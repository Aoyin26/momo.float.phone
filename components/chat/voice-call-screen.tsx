"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { ChatSession, ChatMessage, loadChatMessages, pushChatMessage, getLatestCharacterStateValues } from "@/lib/chat-storage";
import { getStatusRegionConfig, isCustomStatusRegionActive } from "@/lib/chat-status-region";
import type { StateValue } from "@/lib/chat-storage";
import { parseStateValues, mergeStateValues } from "@/lib/state-value-parser";
import { parseAIResponse } from "@/lib/rich-message-parser";
import { generateChatCompletion, flattenCompletionResult, ChatEngineError } from "@/lib/chat-engine";
import { resolveUserIdentity } from "@/lib/settings-storage";
import { cancelFollowUp } from "@/lib/follow-up-service";
import { createSTTSession, type STTSession } from "@/lib/stt-service";
import { resolveVoiceConfig, synthesizeSpeech, playAudioBlob, playAudioBlobViaMediaElement, setCallAudioSessionActive } from "@/lib/tts-service";
import { isCallRecordingSupported, resolveCloudSttConfig } from "@/lib/stt-cloud";
import { useHoldToTalk } from "./use-hold-to-talk";
import { suspendKeepAliveForCall, resumeKeepAliveAfterCall } from "@/lib/use-weixin-bridge";
import { BilingualTextBlock } from "./message-bubble";
import { splitBilingualText } from "@/lib/bilingual-text";
import type { Character } from "@/lib/character-types";
import { useCallKeyboardOffsetStyle } from "./use-call-keyboard-offset";
import { CallSttWarningDialog, hideCallSttWarningPermanently, isCallSttWarningHidden } from "./call-stt-warning-dialog";
import { isAndroidBrowser, isIOSDevice } from "./voice-input-platform";
import { CallVolumeControl } from "./call-volume-control";
import { startIncomingCallVibration } from "@/lib/call-vibration";
import { useCallScreenSounds } from "@/lib/chat-sound";

// ── Types ───────────────────────────────────────────

type CallState =
    | "CONNECTING"
    | "IDLE"
    | "USER_SPEAKING"
    | "PROCESSING"
    | "AI_SPEAKING"
    | "ENDED";

type SubtitleEntry = {
    id: string;
    role: "user" | "assistant";
    text: string;
};

type VoiceCallScreenProps = {
    session: ChatSession;
    character: Character;
    onEnd: () => void;
    onConnect?: () => void;
    initiator?: "user" | "character";
    /** 通话是否处于缩小的悬浮窗状态：暂停麦克风监听/计时/语音播放，仅显示背景+名字 */
    minimized?: boolean;
    /** 点击左上角返回键：请求缩小为悬浮窗（通话逻辑冻结，不挂断） */
    onMinimize?: () => void;
    /** 点击悬浮窗：请求恢复为全屏通话界面 */
    onRestore?: () => void;
};

function stripBilingualForSpeech(text: string): string {
    return text
        .split("\n")
        .map(line => splitBilingualText(line)?.original || line)
        .join("\n");
}

// ── Component ───────────────────────────────────────

export function VoiceCallScreen({ session, character, onEnd, onConnect, initiator = "user", minimized = false, onMinimize, onRestore }: VoiceCallScreenProps) {
    // iOS 保留 Web Speech 免提 + Web Audio 播放（麦克风会话共存的老方案）；
    // 其余设备改「按住说话 + 云端转写」，播放走媒体元素（音量键可控、无静音拨键坑）。
    // 没配 OpenAI 兼容识别时回落旧行为（安卓=文字输入）。
    const iosDeviceRef = useRef(isIOSDevice());
    const iosDevice = iosDeviceRef.current;
    const holdToTalkRef = useRef(
        !iosDeviceRef.current && isCallRecordingSupported() && resolveCloudSttConfig(session.contactId) !== null,
    );
    const holdToTalk = holdToTalkRef.current;
    const androidTextInputOnlyRef = useRef(isAndroidBrowser() && !holdToTalkRef.current);
    const androidTextInputOnly = androidTextInputOnlyRef.current;
    const playCallAudio = iosDevice ? playAudioBlob : playAudioBlobViaMediaElement;
    const keyboardOffsetStyle = useCallKeyboardOffsetStyle();
    const [callState, setCallState] = useState<CallState>("CONNECTING");
    const hasConnectedRef = useRef(false);
    const [callDuration, setCallDuration] = useState(0);
    const [subtitles, setSubtitles] = useState<SubtitleEntry[]>([]);
    const [interimText, setInterimText] = useState("");
    const [isMuted, setIsMuted] = useState(false);
    const [inputMode, setInputMode] = useState<"voice" | "text">(() => androidTextInputOnly ? "text" : "voice");
    const [typedText, setTypedText] = useState("");
    const [bgImageResolved, setBgImageResolved] = useState<string | null>(null);
    const [showSttWarning, setShowSttWarning] = useState(false);

    // ── 哄睡功能状态 ──
    const [showLullabyModal, setShowLullabyModal] = useState(false);
    const [lullabyPlot, setLullabyPlot] = useState("");
    const [lullabyLength, setLullabyLength] = useState("800");
    const [lullabyRunning, setLullabyRunning] = useState(false);
    const [lullabyAutoHangupMin, setLullabyAutoHangupMin] = useState("30");
    const lullabyAbortRef = useRef(false);
    const autoHangupTimerRef = useRef<NodeJS.Timeout | null>(null);

    const sttRef = useRef<STTSession | null>(null);
    const audioAbortRef = useRef<(() => void) | null>(null);
    const timerRef = useRef<NodeJS.Timeout | null>(null);
    const callStartRef = useRef<number>(0);
    const pausedAtRef = useRef<number | null>(null);
    const minimizedRef = useRef(false);
    const stateRef = useRef<string>("CONNECTING");
    const interimTextRef = useRef<string>("");  // ref 版本，闭包安全
    const sttWarningShownRef = useRef(false);
    const subtitleScrollRef = useRef<HTMLDivElement>(null);
    const messagesRef = useRef<ChatMessage[]>([]);
    const _initUi = resolveUserIdentity(session.contactId, "chat");
    const userNameRef = useRef<string>(_initUi?.name || "你");

    // Keep refs in sync
    useEffect(() => { stateRef.current = callState; }, [callState]);
    useEffect(() => { minimizedRef.current = minimized; }, [minimized]);

    // 缩小为悬浮窗：保持通话活跃（麦克风继续监听、TTS 继续播放），仅隐藏全屏 UI
    // 不再冻结任何音频/识别逻辑，实现真实微信小窗体验

    // 来电等待接听：循环振动（开关在聊天主页，iOS 网页不支持自动无效果）
    // + 来电/致电铃声与挂断音（角色专属提示音优先，其余在"全局聊天信息 → 提示音"）
    useCallScreenSounds({ initiator, callState, session });
    useEffect(() => {
        if (initiator !== "character" || callState !== "CONNECTING") return;
        const stop = startIncomingCallVibration();
        return stop;
    }, [initiator, callState]);

    // Pause WeChat keep-alive while the call holds the mic/audio; restore on exit.
    useEffect(() => {
        suspendKeepAliveForCall();
        return () => { resumeKeepAliveAfterCall(); };
    }, []);

    // 通话音频会话 + 卸载兜底：不经挂断键退出（返回聊天页/切会话/组件被销毁）时，
    // 把识别、在途播放与音频会话全部释放。此前识别的自动重启循环在卸载后条件
    // 恒成立（stateRef 停在 IDLE），会在后台无限自我重启，麦克风永不归还，
    // 整页音频被钉在通话模式（语音条/试听音量巨大且音量键失灵）。
    useEffect(() => {
        setCallAudioSessionActive(true);
        return () => {
            stateRef.current = "ENDED";
            if (sttRef.current) { sttRef.current.abort(); sttRef.current = null; }
            if (audioAbortRef.current) { audioAbortRef.current(); audioAbortRef.current = null; }
            setCallAudioSessionActive(false);
        };
    }, []);
    useEffect(() => { interimTextRef.current = interimText; }, [interimText]);

    const showSttCompatibilityWarning = useCallback(() => {
        if (androidTextInputOnly) {
            setInputMode("text");
            return;
        }
        if (sttWarningShownRef.current || isCallSttWarningHidden()) return;
        sttWarningShownRef.current = true;
        setShowSttWarning(true);
    }, [androidTextInputOnly]);

    const handleNeverShowSttWarning = useCallback(() => {
        hideCallSttWarningPermanently();
        setShowSttWarning(false);
    }, []);

    // Scroll subtitles to bottom on change
    useEffect(() => {
        if (subtitleScrollRef.current) {
            subtitleScrollRef.current.scrollTop = subtitleScrollRef.current.scrollHeight;
        }
    }, [subtitles, interimText]);

    // ── Resolve voiceBackground from IndexedDB ──────

    useEffect(() => {
        if (!session.voiceBackground) {
            setBgImageResolved(null);
            return;
        }
        if (session.voiceBackground.startsWith("data:") || session.voiceBackground.startsWith("http")) {
            setBgImageResolved(session.voiceBackground);
            return;
        }
        // IndexedDB ID
        import("@/lib/chat-asset-storage").then(({ getChatImageFromIndexedDB }) => {
            getChatImageFromIndexedDB(session.voiceBackground!).then(dataUrl => {
                if (dataUrl) setBgImageResolved(dataUrl);
            });
        });
    }, [session.voiceBackground]);

    // ── Call timer ───────────────────────────────────

    useEffect(() => {
        if (callState === "CONNECTING" || callState === "ENDED") return;

        if (!callStartRef.current) {
            callStartRef.current = Date.now();
        }

        // 小窗模式下计时器也继续推进（通话不冻结）
        timerRef.current = setInterval(() => {
            setCallDuration(Math.floor((Date.now() - callStartRef.current) / 1000));
        }, 1000);

        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
        };
    }, [callState, minimized]);

    // ── Connecting animation (3s fake dial) ─────────

    useEffect(() => {
        cancelFollowUp(session.id);

        // Resolve user name
        const ui = resolveUserIdentity(session.contactId, "chat");
        userNameRef.current = ui?.name || "你";

        // Load existing messages for context
        messagesRef.current = loadChatMessages(session.id);

        // Insert system message (skip if already exists from strict mode remount)
        const lastMsg = messagesRef.current[messagesRef.current.length - 1];
        const initRole = initiator === "character" ? "assistant" : "user";
        if (!lastMsg || !(lastMsg.content.includes("发起了语音通话"))) {
            const callMsg = initiator === "character"
                ? `[我向${userNameRef.current}发起了语音通话]`
                : `[我向${character.name}发起了语音通话]`;
            const sysMsg = pushChatMessage({
                sessionId: session.id,
                role: initRole,
                content: callMsg,
            });
            messagesRef.current = [...messagesRef.current, sysMsg];
        }

        // User-initiated: auto-connect after 3s fake dial
        // Character-initiated: wait for user to accept
        let connectTimer: NodeJS.Timeout | undefined;
        if (initiator !== "character") {
            connectTimer = setTimeout(() => {
                setCallState("IDLE");
            }, 3000);
        }

        return () => {
            if (connectTimer) clearTimeout(connectTimer);
            if (timerRef.current) clearInterval(timerRef.current);
        };
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Track first connect
    useEffect(() => {
        if (callState !== "CONNECTING" && !hasConnectedRef.current) {
            hasConnectedRef.current = true;
        }
    }, [callState]);

    // ── Format time MM:SS ───────────────────────────

    const formatTime = (seconds: number) => {
        const m = Math.floor(seconds / 60);
        const s = seconds % 60;
        return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
    };

    // ── State label ─────────────────────────────────

    const stateLabel = (): string => {
        switch (callState) {
            case "CONNECTING": return initiator === "character" ? "来电..." : "正在呼叫...";
            case "IDLE": return isMuted ? "已静音" : "通话中";
            case "USER_SPEAKING": return "正在聆听...";
            case "PROCESSING": return "对方正在思考...";
            case "AI_SPEAKING": return "对方正在说话...";
            case "ENDED": return "通话已结束";
        }
    };

    // ── AI response processing (same logic as chat-room) ──

    const processAIResponse = useCallback((aiResponseText: string): { cleanParts: string[]; stateValues: StateValue[] } => {
        // Use shared parseAIResponse for full rich media support (stickers, quotes, etc.)
        const previousState = getLatestCharacterStateValues(session.contactId);

        const { parts, stateValues, freshStateValues, statusPanel, innerMonologue } = parseAIResponse(aiResponseText, previousState);

        // 自定义状态栏渲染戳：不盖的话 custom 模式下 [状态栏] 原文按 markdown 渲染，看着像掉格式
        const statusRegionMode = statusPanel && isCustomStatusRegionActive(getStatusRegionConfig(session.id))
            ? ("custom" as const)
            : undefined;

        // Filter out non-chat action types (voice_call, video_call, poke, etc.)
        const chatParts = parts.filter(p =>
            !p.mediaType || !["voice_call", "video_call", "poke", "accept_red_packet", "decline_red_packet", "accept_transfer", "decline_transfer", "accept_payment_request", "decline_payment_request"].includes(p.mediaType)
        );

        // Save messages to storage
        if (chatParts.length === 0 && (statusPanel || innerMonologue)) {
            const aiMsg = pushChatMessage({
                sessionId: session.id,
                role: "assistant",
                content: "",
                statusPanel,
                statusRegionMode,
                innerMonologue,
                stateValues: stateValues.length > 0 ? stateValues : undefined,
                freshStateValues,
            });
            messagesRef.current = [...messagesRef.current, aiMsg];
        } else {
            const newMsgs = chatParts.map((part, idx) =>
                pushChatMessage({
                    sessionId: session.id,
                    role: "assistant",
                    content: part.content,
                    mediaType: part.mediaType,
                    mediaData: part.mediaData,
                    statusPanel: idx === 0 && statusPanel ? statusPanel : undefined,
                    statusRegionMode: idx === 0 && statusPanel ? statusRegionMode : undefined,
                    innerMonologue: idx === 0 && innerMonologue ? innerMonologue : undefined,
                    stateValues: idx === 0 && stateValues.length > 0 ? stateValues : undefined,
                    freshStateValues: idx === 0 ? freshStateValues : undefined,
                })
            );
            messagesRef.current = [...messagesRef.current, ...newMsgs];
        }

        // Return clean text parts for TTS (exclude rich media content)
        const cleanParts = chatParts
            .filter(p => !p.mediaType && p.content.trim())
            .map(p => p.content);

        return { cleanParts, stateValues };
    }, [session.id, session.contactId]);

    // ── Full conversation turn ──────────────────────

    const runConversationTurn = useCallback(async (userText?: string) => {
        // 1. Save user message (skip for initial greeting)
        if (userText) {
            const userMsg = pushChatMessage({
                sessionId: session.id,
                role: "user",
                content: userText,
            });
            messagesRef.current = [...messagesRef.current, userMsg];

            // Add user subtitle
            setSubtitles(prev => [...prev, { id: userMsg.id, role: "user", text: userText }]);
        }

        // 2. Switch to PROCESSING
        setCallState("PROCESSING");
        setInterimText("");

        try {
            // 3. Generate AI response
            const aiResponseText = flattenCompletionResult(await generateChatCompletion(session, messagesRef.current, {
                appTags: ["chat", "voice"],
            }));

            // Bail if call ended during generation
            if (stateRef.current === "ENDED") return;

            // 4. Process response
            const { cleanParts } = processAIResponse(aiResponseText);
            const displayText = cleanParts.join("\n");
            const speechText = stripBilingualForSpeech(displayText);

            if (!displayText) {
                setCallState("IDLE");
                return;
            }

            // 5. Add AI subtitle
            const subtitleId = `ai-${Date.now()}`;
            setSubtitles(prev => [...prev, { id: subtitleId, role: "assistant", text: displayText }]);

            // 6. TTS（小窗模式下也继续播放语音，实现真实微信小窗体验）
            setCallState("AI_SPEAKING");

            const voiceConfig = resolveVoiceConfig(session.contactId);
            if (voiceConfig) {
                try {
                    const audioBlob = await synthesizeSpeech(speechText, voiceConfig);
                    if (stateRef.current === "ENDED") return;

                    if (audioBlob) {
                        const { promise, abort } = playCallAudio(audioBlob);
                        audioAbortRef.current = abort;
                        await promise;
                        audioAbortRef.current = null;
                    }
                } catch (e) {
                    console.warn("[VoiceCall] TTS failed:", e);
                }
            }

            if (stateRef.current !== "ENDED") {
                setCallState("IDLE");
            }
        } catch (error: any) {
            console.error("[VoiceCall] Error:", error);
            if (stateRef.current !== "ENDED") {
                setSubtitles(prev => [...prev, {
                    id: `err-${Date.now()}`,
                    role: "assistant",
                    text: `⚠️ ${error?.message || "发送失败"}`,
                }]);
                setCallState("IDLE");
            }
        }
    }, [session, processAIResponse, playCallAudio]);

    // ── 哄睡：按大纲生成长文 → 分段 TTS 朗读 → 写入记忆 ──

    const startLullaby = useCallback(async () => {
        if (lullabyRunning) return;
        const voiceConfig = resolveVoiceConfig(session.contactId);
        if (!voiceConfig) {
            setSubtitles(prev => [...prev, { id: `lullaby-err-${Date.now()}`, role: "assistant", text: "⚠️ 当前角色未绑定语音音色，无法哄睡" }]);
            return;
        }

        setShowLullabyModal(false);
        setLullabyRunning(true);
        lullabyAbortRef.current = false;

        // 自动挂断定时器
        const hangupMin = Math.max(1, parseInt(lullabyAutoHangupMin) || 30);
        if (autoHangupTimerRef.current) clearTimeout(autoHangupTimerRef.current);
        autoHangupTimerRef.current = setTimeout(() => {
            lullabyAbortRef.current = true;
        }, hangupMin * 60 * 1000);

        const targetLen = Math.max(200, parseInt(lullabyLength) || 800);
        const plotHint = lullabyPlot.trim() || "温柔地哄对方入睡，可以讲一个舒缓的小故事或者轻声安慰";

        try {
            setCallState("PROCESSING");
            setSubtitles(prev => [...prev, { id: `lullaby-start-${Date.now()}`, role: "assistant", text: "🌙 哄睡模式已开启..." }]);

            const lullabyPrompt = `你正在通话中哄对方睡觉。请根据以下大纲/要求，用温柔、舒缓、适合入睡的语气，生成一段约${targetLen}字的哄睡内容。不要输出任何标签、状态栏或内心独白，只输出纯正文。语速放慢，句子之间留有停顿感。\n\n哄睡大纲：${plotHint}`;

            const lullabyUserMsg = pushChatMessage({
                sessionId: session.id,
                role: "user",
                content: `[哄睡模式] ${plotHint}`,
            });
            messagesRef.current = [...messagesRef.current, lullabyUserMsg];

            const aiText = flattenCompletionResult(await generateChatCompletion(session, [
                ...messagesRef.current.slice(-20),
                { id: `lullaby_sys_${Date.now()}`, sessionId: session.id, role: "user" as const, content: lullabyPrompt, status: "sent" as const, createdAt: new Date().toISOString() },
            ], { appTags: ["chat", "voice", "lullaby"] }));

            if (stateRef.current === "ENDED" || lullabyAbortRef.current) return;

            // 写入聊天记忆
            const aiMsg = pushChatMessage({
                sessionId: session.id,
                role: "assistant",
                content: aiText,
            });
            messagesRef.current = [...messagesRef.current, aiMsg];

            // 分段朗读（按句号/感叹号/问号/省略号/换行切分）
            const segments = aiText.split(/(?<=[。！？…\n])\s*/).filter(s => s.trim());
            setCallState("AI_SPEAKING");

            for (let i = 0; i < segments.length; i++) {
                if (stateRef.current === "ENDED" || lullabyAbortRef.current) break;
                const seg = segments[i].trim();
                if (!seg) continue;

                setSubtitles(prev => [...prev, { id: `lullaby-${Date.now()}-${i}`, role: "assistant", text: seg }]);

                const speechText = stripBilingualForSpeech(seg);
                try {
                    const blob = await synthesizeSpeech(speechText, voiceConfig);
                    if (stateRef.current === "ENDED" || lullabyAbortRef.current) break;
                    if (blob) {
                        const { promise, abort } = playCallAudio(blob);
                        audioAbortRef.current = abort;
                        await promise;
                        audioAbortRef.current = null;
                    }
                } catch (e) {
                    console.warn("[Lullaby] TTS segment failed:", e);
                }
            }

            if (stateRef.current !== "ENDED") {
                setSubtitles(prev => [...prev, { id: `lullaby-end-${Date.now()}`, role: "assistant", text: "🌙 哄睡内容已读完，晚安～" }]);
                setCallState("IDLE");
            }
        } catch (error: any) {
            console.error("[Lullaby] Error:", error);
            if (stateRef.current !== "ENDED") {
                setSubtitles(prev => [...prev, { id: `lullaby-err2-${Date.now()}`, role: "assistant", text: `⚠️ 哄睡生成失败: ${error?.message || "未知错误"}` }]);
                setCallState("IDLE");
            }
        } finally {
            setLullabyRunning(false);
            if (autoHangupTimerRef.current) { clearTimeout(autoHangupTimerRef.current); autoHangupTimerRef.current = null; }
        }
    }, [session, lullabyPlot, lullabyLength, lullabyAutoHangupMin, lullabyRunning, playCallAudio, processAIResponse]);

    // ── Auto-listen: 进入 IDLE 自动开始监听 ────────

    const startListening = useCallback(() => {
        if (holdToTalk) return; // 按住说话模式不用 Web Speech 自动监听
        if (androidTextInputOnly) {
            setInputMode("text");
            return;
        }
        if (sttRef.current) {
            sttRef.current.abort();
            sttRef.current = null;
        }
        setInterimText("");
        interimTextRef.current = "";

        const stt = createSTTSession({
            onInterim: (text) => {
                setInterimText(text);
                interimTextRef.current = text;
                // 有中间结果 → 切到 USER_SPEAKING
                if (stateRef.current === "IDLE") {
                    setCallState("USER_SPEAKING");
                }
            },
            onFinal: (text) => {
                sttRef.current = null;
                if (text.trim()) {
                    runConversationTurn(text.trim());
                } else {
                    setInterimText("");
                    setCallState("IDLE");
                }
            },
            onError: (error) => {
                console.warn("[VoiceCall] STT error:", error);
                sttRef.current = null;
                setInterimText("");
                showSttCompatibilityWarning();
                // 严重错误，回到 IDLE（会触发重新监听）
                if (stateRef.current === "USER_SPEAKING" || stateRef.current === "IDLE") {
                    setCallState("IDLE");
                }
            },
            onNoSpeech: () => {
                // 没检测到语音 → 静默重新开始监听
                sttRef.current = null;
                showSttCompatibilityWarning();
                if (stateRef.current === "IDLE" || stateRef.current === "USER_SPEAKING") {
                    // 短暂延迟后重启，避免快速循环
                    setTimeout(() => {
                        if (stateRef.current === "IDLE") {
                            startListening();
                        }
                    }, 300);
                }
            },
            onEnd: () => {
                // 没有 finalText 也没有 no-speech → 用 interimRef 兜底
                sttRef.current = null;
                if (stateRef.current === "USER_SPEAKING" || stateRef.current === "IDLE") {
                    const fallback = interimTextRef.current;
                    if (fallback.trim()) {
                        runConversationTurn(fallback.trim());
                    } else {
                        setInterimText("");
                        setCallState("IDLE");
                    }
                }
            },
        }, "zh-CN");

        sttRef.current = stt;

        if (stt.isSupported) {
            stt.start();
        } else {
            sttRef.current = null;
            showSttCompatibilityWarning();
        }
    }, [androidTextInputOnly, holdToTalk, runConversationTurn, session.contactId, showSttCompatibilityWarning]);

    // IDLE 时自动开启监听（按住说话模式无自动监听，识别只在按住期间发生）
    useEffect(() => {
        if (holdToTalk) return;
        if (inputMode === "text" && sttRef.current) {
            sttRef.current.abort();
            sttRef.current = null;
            setInterimText("");
        }
        if (!androidTextInputOnly && inputMode === "voice" && callState === "IDLE" && !isMuted && !minimized) {
            // 短暂延迟让 UI 过渡完成
            const timer = setTimeout(() => {
                if (stateRef.current === "IDLE" && !minimizedRef.current) {
                    startListening();
                }
            }, 500);
            return () => clearTimeout(timer);
        }
        // 静音时停止监听
        if (isMuted && sttRef.current) {
            sttRef.current.abort();
            sttRef.current = null;
        }
    }, [androidTextInputOnly, holdToTalk, callState, isMuted, inputMode, minimized, startListening]);

    const handleInputModeToggle = useCallback(() => {
        if (androidTextInputOnly) {
            if (sttRef.current) {
                sttRef.current.abort();
                sttRef.current = null;
            }
            setInterimText("");
            if (stateRef.current === "USER_SPEAKING") setCallState("IDLE");
            setInputMode("text");
            return;
        }
        if (inputMode === "voice") {
            if (sttRef.current) {
                sttRef.current.abort();
                sttRef.current = null;
            }
            setInterimText("");
            if (stateRef.current === "USER_SPEAKING") setCallState("IDLE");
            setInputMode("text");
        } else {
            setInputMode("voice");
        }
    }, [androidTextInputOnly, inputMode]);

    const handleTextSubmit = useCallback(() => {
        const text = typedText.trim();
        if (!text || callState !== "IDLE") return;
        if (sttRef.current) {
            sttRef.current.abort();
            sttRef.current = null;
        }
        setTypedText("");
        runConversationTurn(text);
    }, [typedText, callState, runConversationTurn]);

    // 输入框左侧的"重回"键：不发送新内容，直接让对方基于当前上下文重新回复一次
    const handleRegenerate = useCallback(() => {
        if (callState !== "IDLE") return;
        if (sttRef.current) {
            sttRef.current.abort();
            sttRef.current = null;
        }
        runConversationTurn();
    }, [callState, runConversationTurn]);

    // 按住说话（非 iOS）：按下录音，松开转写后走对话轮
    const holdInput = useHoldToTalk({
        characterId: session.contactId,
        canStart: () => stateRef.current === "IDLE",
        onRecordingStart: () => {
            setInterimText("");
            if (stateRef.current === "IDLE") setCallState("USER_SPEAKING");
        },
        onTranscribeStart: () => {
            if (stateRef.current === "USER_SPEAKING") setCallState("PROCESSING");
        },
        onTranscript: (text) => { void runConversationTurn(text); },
        onError: () => {
            if (stateRef.current === "USER_SPEAKING" || stateRef.current === "PROCESSING") {
                setCallState("IDLE");
            }
        },
    });

    // ── Hangup ──────────────────────────────────────

    const handleHangup = useCallback(() => {
        setCallState("ENDED");

        // Stop any ongoing STT
        if (sttRef.current) {
            sttRef.current.abort();
            sttRef.current = null;
        }

        // Stop any ongoing audio playback
        if (audioAbortRef.current) {
            audioAbortRef.current();
            audioAbortRef.current = null;
        }

        // Stop browser TTS
        if (window.speechSynthesis) {
            window.speechSynthesis.cancel();
        }

        const endMsg = pushChatMessage({
            sessionId: session.id,
            role: "user",
            content: `[我挂断了语音通话]`,
            mediaData: { callDuration: formatTime(callDuration) },
        });
        messagesRef.current = [...messagesRef.current, endMsg];

        // Delay then close
        setTimeout(() => onEnd(), 1500);
    }, [session.id, callDuration, onEnd]);

    // ── Render ──────────────────────────────────────

    if (minimized) {
        return (
            <div
                className="call-mini-window"
                style={{ backgroundImage: `url(${bgImageResolved || character.avatar || ""})`, cursor: "pointer" }}
                aria-label={`返回与${character.name}的语音通话`}
                title="点击返回通话"
            >
                <span className="call-mini-window-overlay" />
                <button
                    type="button"
                    onClick={onRestore}
                    style={{ position: "absolute", inset: 0, background: "transparent", border: "none", cursor: "pointer", zIndex: 1 }}
                    aria-label="恢复全屏通话"
                />
                <span className="call-mini-window-name">{character.name}</span>
                <span style={{
                    position: "absolute", bottom: 4, left: 0, right: 0,
                    textAlign: "center", fontSize: 10, color: "#fff", opacity: 0.85,
                    textShadow: "0 1px 3px rgba(0,0,0,0.6)", zIndex: 2, pointerEvents: "none",
                }}>
                    {formatTime(callDuration)}
                    {callState === "AI_SPEAKING" ? " 🔊" : callState === "PROCESSING" ? " 💭" : ""}
                </span>
                <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); handleHangup(); }}
                    style={{
                        position: "absolute", top: -6, right: -6, zIndex: 3,
                        width: 22, height: 22, borderRadius: "50%",
                        background: "#e53e3e", border: "2px solid #fff",
                        display: "flex", alignItems: "center", justifyContent: "center",
                        cursor: "pointer", padding: 0, lineHeight: 1,
                    }}
                    aria-label="挂断"
                >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#fff" strokeWidth="3" strokeLinecap="round"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
                </button>
            </div>
        );
    }

    return (
        <div
            className="absolute inset-0 z-[100] flex flex-col text-white overflow-hidden call-bg-default call-keyboard-shift"
            style={bgImageResolved ? { ...keyboardOffsetStyle, background: `url(${bgImageResolved}) center/cover no-repeat` } : keyboardOffsetStyle}
        >
            {/* Dark overlay for readability */}
            <div
                className="call-overlay absolute inset-0 z-0"
                {...(bgImageResolved ? { "data-has-image": "" } : {})}
            />

            <CallVolumeControl />

            {onMinimize && callState !== "ENDED" && (
                <button
                    type="button"
                    className="call-back-btn"
                    onClick={onMinimize}
                    aria-label="缩小通话"
                    title="缩小通话"
                >
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M15 18l-6-6 6-6" />
                    </svg>
                </button>
            )}

            {/* Content wrapper — force white text so themes don't override call UI */}
            <div className="voicecall-controls gcall-body">
                {/* Top: Duration + Status */}
                <div className="gcall-topbar">
                    <div className="gcall-topbar-title">
                        {character.name}
                    </div>
                    <div
                        className="gcall-topbar-sub"
                        {...(callState === "CONNECTING" || callState === "PROCESSING" ? { "data-anim": "" } : {})}
                    >
                        {callState !== "CONNECTING" && callState !== "ENDED" ? `${formatTime(callDuration)} · ` : ""}
                        {stateLabel()}
                    </div>
                </div>

                {/* Center: Avatar + connecting ring */}
                <div className="flex-none flex justify-center items-center pt-[30px] pb-5">
                    <div className="relative flex items-center justify-center">
                        <div
                            className="voicecall-avatar"
                            {...(callState === "AI_SPEAKING" ? { "data-speaking": "" } : {})}
                        >
                            {character.avatar ? (
                                <img
                                    src={character.avatar}
                                    alt={character.name}
                                    className="w-full h-full object-cover"
                                />
                            ) : (
                                <span className="ts-48 text-[var(--c-icon)]">
                                    {character.name?.[0] || "?"}
                                </span>
                            )}
                        </div>
                        {callState === "CONNECTING" && (
                            <>
                                <div
                                    className="absolute w-[160px] h-[160px] rounded-full pointer-events-none"
                                    style={{
                                        border: "2px solid rgba(255,255,255,0.2)",
                                        animation: "voicecall-ring 1.5s ease-out infinite",
                                    }}
                                />
                                <div
                                    className="absolute w-[160px] h-[160px] rounded-full pointer-events-none"
                                    style={{
                                        border: "2px solid rgba(255,255,255,0.2)",
                                        animation: "voicecall-ring 1.5s ease-out infinite 0.5s",
                                    }}
                                />
                            </>
                        )}
                    </div>
                </div>

                <div className="text-center ts-18 font-semibold mb-2">
                    {character.name}
                </div>

                {/* Subtitle area — top fade via mask */}
                <div
                    ref={subtitleScrollRef}
                    className="voicecall-subtitle-mask flex-1 min-h-0 overflow-auto px-5 py-[10px] flex flex-col gap-2 relative"
                    {...(inputMode === "text" && callState !== "CONNECTING" && callState !== "ENDED" ? { "data-text-input": "" } : {})}
                >
                    {subtitles.map((sub) => (
                        <div
                            key={sub.id}
                            className="call-subtitle"
                            data-role={sub.role}
                        >
                            <BilingualTextBlock text={sub.text} mode="plain" className="call-subtitle-bilingual" defaultExpanded={session.collapseBilingualTranslation !== false ? false : true} />
                        </div>
                    ))}

                    {/* Interim STT text */}
                    {interimText && callState === "USER_SPEAKING" && (
                        <div className="call-subtitle" data-interim="">
                            {interimText}
                        </div>
                    )}
                </div>

                {inputMode === "text" && callState !== "CONNECTING" && callState !== "ENDED" && (
                    <form
                        className="call-text-input-panel voicecall-text-input-panel call-text-input-row"
                        onSubmit={(e) => {
                            e.preventDefault();
                            handleTextSubmit();
                        }}
                    >
                        <button
                            type="button"
                            onClick={handleRegenerate}
                            className="call-regenerate-btn"
                            disabled={callState !== "IDLE"}
                            aria-label="让对方重新回复"
                            title="让对方重新回复"
                        >
                            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M23 4v6h-6" />
                                <path d="M1 20v-6h6" />
                                <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
                            </svg>
                        </button>
                        <div className="call-text-input-shell">
                            <input
                                value={typedText}
                                onChange={e => setTypedText(e.target.value)}
                                className="call-text-input"
                                placeholder={callState === "IDLE" ? "输入你想说的话..." : "稍等对方说完..."}
                                disabled={callState !== "IDLE"}
                            />
                            <button
                                type="submit"
                                className="call-text-send-btn"
                                disabled={!typedText.trim() || callState !== "IDLE"}
                                aria-label="发送"
                            >
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.3" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M12 19V5" />
                                    <path d="M5 12l7-7 7 7" />
                                </svg>
                            </button>
                        </div>
                    </form>
                )}

                {/* 按住说话提示/错误行 */}
                {holdToTalk && inputMode === "voice" && callState !== "CONNECTING" && callState !== "ENDED" && (
                    <div className="text-center ts-12 opacity-80 px-5">
                        {holdInput.recState === "recording" ? "松开发送"
                            : holdInput.recState === "transcribing" ? "识别中…"
                            : holdInput.error || "按住下方麦克风说话"}
                    </div>
                )}

                {/* Bottom controls */}
                <div
                    className="flex justify-center items-center gap-[40px] p-5"
                    style={{ paddingBottom: "max(30px, env(safe-area-inset-bottom))" }}
                >
                    {callState !== "ENDED" && callState !== "CONNECTING" ? holdToTalk ? (
                        <>
                            {/* 输入方式切换（按住说话模式不需要持续开麦，静音位改放 Aa 切换） */}
                            <button
                                onClick={handleInputModeToggle}
                                className="ui-call-btn ui-call-btn-muted"
                                aria-label={inputMode === "voice" ? "切换到文字输入" : "切换到语音输入"}
                            >
                                {inputMode === "voice" ? (
                                    <span className="ui-call-input-text-icon">Aa</span>
                                ) : (
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                                        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                                        <line x1="12" y1="19" x2="12" y2="22" />
                                    </svg>
                                )}
                            </button>

                            {/* 按住说话主按钮（文字模式下点按切回语音） */}
                            <button
                                className="ui-call-mic ui-call-mic-lg"
                                style={{ touchAction: "none" }}
                                data-state={
                                    inputMode === "text" ? "text"
                                        : holdInput.recState === "recording" ? "speaking"
                                        : callState === "IDLE" ? "idle"
                                        : "busy"
                                }
                                aria-label={inputMode === "text" ? "切换到语音输入" : "按住说话"}
                                title={inputMode === "text" ? "切换到语音输入" : "按住说话"}
                                {...(inputMode === "voice" ? holdInput.pressHandlers : { onClick: handleInputModeToggle })}
                            >
                                {inputMode === "text" ? (
                                    <span className="ui-call-input-text-icon">Aa</span>
                                ) : (
                                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                                        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                                        <line x1="12" y1="19" x2="12" y2="22" />
                                    </svg>
                                )}
                            </button>

                            {/* Hangup */}
                            <button
                                onClick={handleHangup}
                                className="ui-call-btn ui-call-btn-danger"
                                aria-label="挂断"
                            >
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M10.68 13.31a16 16 0 0 0 3.41 2.6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7 2 2 0 0 1 1.72 2v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.42 19.42 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91" />
                                    <line x1="23" y1="1" x2="1" y2="23" />
                                </svg>
                            </button>
                        </>
                    ) : androidTextInputOnly ? (
                        <button
                            onClick={handleHangup}
                            className="ui-call-btn ui-call-btn-danger"
                            aria-label="挂断"
                        >
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M10.68 13.31a16 16 0 0 0 3.41 2.6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7 2 2 0 0 1 1.72 2v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.42 19.42 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91" />
                                <line x1="23" y1="1" x2="1" y2="23" />
                            </svg>
                        </button>
                    ) : (
                        <>
                            {/* Mute button */}
                            <button
                                onClick={() => setIsMuted(!isMuted)}
                                className="ui-call-btn ui-call-btn-muted"
                                {...(isMuted ? { "data-checked": "" } : {})}
                            >
                                {isMuted ? (
                                    /* Muted: mic with diagonal */
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <line x1="1" y1="1" x2="23" y2="23" />
                                        <path d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6" />
                                        <path d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2c0 .76-.13 1.48-.35 2.15" />
                                        <line x1="12" y1="19" x2="12" y2="23" /><line x1="8" y1="23" x2="16" y2="23" />
                                    </svg>
                                ) : (
                                    /* Active mic */
                                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                                        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                                        <line x1="12" y1="19" x2="12" y2="22" />
                                    </svg>
                                )}
                            </button>

                            {/* Mic button — input mode toggle with voice-state indicator */}
                            <button
                                onClick={handleInputModeToggle}
                                className="ui-call-mic ui-call-mic-lg"
                                data-state={
                                    inputMode === "text" ? "text"
                                        : callState === "USER_SPEAKING" ? "speaking"
                                        : callState === "IDLE" ? (isMuted ? "idle-muted" : "idle")
                                        : "busy"
                                }
                                aria-label={androidTextInputOnly ? "文字输入" : inputMode === "voice" ? "切换到文字输入" : "切换到语音输入"}
                                title={androidTextInputOnly ? "安卓浏览器使用文字输入" : inputMode === "voice" ? "切换到文字输入" : "切换到语音输入"}
                            >
                                {inputMode === "text" ? (
                                    <span className="ui-call-input-text-icon">Aa</span>
                                ) : (
                                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                                        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                                        <line x1="12" y1="19" x2="12" y2="22" />
                                    </svg>
                                )}
                            </button>

                            {/* Hangup button */}
                            <button
                                onClick={handleHangup}
                                className="ui-call-btn ui-call-btn-danger"
                            >
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M10.68 13.31a16 16 0 0 0 3.41 2.6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7 2 2 0 0 1 1.72 2v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.42 19.42 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91" />
                                    <line x1="23" y1="1" x2="1" y2="23" />
                                </svg>
                            </button>
                        </>
                    ) : callState === "CONNECTING" && initiator === "character" ? (
                        /* Incoming call: accept + decline */
                        <>
                            <button
                                onClick={() => {
                                    pushChatMessage({
                                        sessionId: session.id,
                                        role: "user",
                                        content: `[我拒绝了语音通话]`,
                                    });
                                    onEnd();
                                }}
                                className="ui-call-btn ui-call-btn-danger"
                            >
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M10.68 13.31a16 16 0 0 0 3.41 2.6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7 2 2 0 0 1 1.72 2v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.42 19.42 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91" />
                                    <line x1="23" y1="1" x2="1" y2="23" />
                                </svg>
                            </button>
                            <button
                                onClick={() => setCallState("IDLE")}
                                className="ui-call-btn ui-call-btn-success"
                            >
                                {/* Phone pick-up icon */}
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z" />
                                </svg>
                            </button>
                        </>
                    ) : callState === "CONNECTING" ? (
                        /* User-initiated: show cancel only */
                        <button
                            onClick={() => {
                                pushChatMessage({
                                    sessionId: session.id,
                                    role: "user",
                                    content: `[我取消了语音通话]`,
                                });
                                onEnd();
                            }}
                            className="ui-call-btn ui-call-btn-danger"
                        >
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M10.68 13.31a16 16 0 0 0 3.41 2.6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7 2 2 0 0 1 1.72 2v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.42 19.42 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91" />
                                <line x1="23" y1="1" x2="1" y2="23" />
                            </svg>
                        </button>
                    ) : (
                        /* ENDED state: show nothing, will auto-close */
                        <div className="ts-14 opacity-70">通话已结束</div>
                    )}
                </div>
            </div>

            {!androidTextInputOnly && showSttWarning && (
                <CallSttWarningDialog
                    onClose={() => setShowSttWarning(false)}
                    onNeverShow={handleNeverShowSttWarning}
                />
            )}

            {/* 哄睡按钮：通话中且非哄睡运行时显示 */}
            {callState !== "CONNECTING" && callState !== "ENDED" && !lullabyRunning && (
                <button
                    onClick={() => setShowLullabyModal(true)}
                    style={{
                        position: "absolute", top: 70, right: 16,
                        background: "rgba(0,0,0,0.35)", backdropFilter: "blur(8px)",
                        border: "none", borderRadius: 20, padding: "6px 14px",
                        color: "#fff", fontSize: "calc(13px*var(--app-text-scale,1))",
                        cursor: "pointer", zIndex: 10, display: "flex", alignItems: "center", gap: 5,
                    }}
                    aria-label="哄睡模式"
                >
                    🌙 哄睡
                </button>
            )}
            {lullabyRunning && (
                <button
                    onClick={() => { lullabyAbortRef.current = true; if (audioAbortRef.current) audioAbortRef.current(); }}
                    style={{
                        position: "absolute", top: 70, right: 16,
                        background: "rgba(180,60,60,0.7)", backdropFilter: "blur(8px)",
                        border: "none", borderRadius: 20, padding: "6px 14px",
                        color: "#fff", fontSize: "calc(13px*var(--app-text-scale,1))",
                        cursor: "pointer", zIndex: 10,
                    }}
                    aria-label="停止哄睡"
                >
                    ⏹ 停止哄睡
                </button>
            )}

            {/* 哄睡配置弹窗 */}
            {showLullabyModal && (
                <div style={{ position: "absolute", inset: 0, zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center" }}>
                    <div onClick={() => setShowLullabyModal(false)} style={{ position: "absolute", inset: 0, background: "rgba(0,0,0,0.5)" }} />
                    <div style={{
                        position: "relative", width: "85%", maxWidth: 340,
                        background: "var(--c-page-bg, #1a1a2e)", borderRadius: 16,
                        padding: 20, color: "var(--c-text, #e0e0e0)",
                        boxShadow: "0 8px 32px rgba(0,0,0,0.4)",
                    }}>
                        <div style={{ fontSize: "calc(16px*var(--app-text-scale,1))", fontWeight: 600, marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}>
                            🌙 哄睡模式
                        </div>

                        <label style={{ fontSize: "calc(13px*var(--app-text-scale,1))", opacity: 0.7, display: "block", marginBottom: 4 }}>哄睡大纲 / 想让角色说什么</label>
                        <textarea
                            value={lullabyPlot}
                            onChange={e => setLullabyPlot(e.target.value)}
                            placeholder="例：讲一个森林里小鹿散步的故事，温柔安静的氛围"
                            rows={3}
                            style={{
                                width: "100%", boxSizing: "border-box", borderRadius: 8,
                                border: "1px solid rgba(255,255,255,0.15)", background: "rgba(255,255,255,0.08)",
                                color: "inherit", padding: "8px 10px", fontSize: "calc(13px*var(--app-text-scale,1))",
                                resize: "vertical",
                            }}
                        />

                        <div style={{ display: "flex", gap: 12, marginTop: 12 }}>
                            <div style={{ flex: 1 }}>
                                <label style={{ fontSize: "calc(12px*var(--app-text-scale,1))", opacity: 0.7, display: "block", marginBottom: 4 }}>字数</label>
                                <input
                                    type="number"
                                    value={lullabyLength}
                                    onChange={e => setLullabyLength(e.target.value)}
                                    min={200} max={10000} step={100}
                                    style={{
                                        width: "100%", boxSizing: "border-box", borderRadius: 8,
                                        border: "1px solid rgba(255,255,255,0.15)", background: "rgba(255,255,255,0.08)",
                                        color: "inherit", padding: "8px 10px", fontSize: "calc(13px*var(--app-text-scale,1))",
                                    }}
                                />
                            </div>
                            <div style={{ flex: 1 }}>
                                <label style={{ fontSize: "calc(12px*var(--app-text-scale,1))", opacity: 0.7, display: "block", marginBottom: 4 }}>自动挂断(分钟)</label>
                                <input
                                    type="number"
                                    value={lullabyAutoHangupMin}
                                    onChange={e => setLullabyAutoHangupMin(e.target.value)}
                                    min={1} max={180} step={5}
                                    style={{
                                        width: "100%", boxSizing: "border-box", borderRadius: 8,
                                        border: "1px solid rgba(255,255,255,0.15)", background: "rgba(255,255,255,0.08)",
                                        color: "inherit", padding: "8px 10px", fontSize: "calc(13px*var(--app-text-scale,1))",
                                    }}
                                />
                            </div>
                        </div>

                        <div style={{ display: "flex", gap: 10, marginTop: 18 }}>
                            <button
                                onClick={() => setShowLullabyModal(false)}
                                style={{
                                    flex: 1, padding: "10px 0", borderRadius: 10, border: "1px solid rgba(255,255,255,0.2)",
                                    background: "transparent", color: "inherit", fontSize: "calc(14px*var(--app-text-scale,1))", cursor: "pointer",
                                }}
                            >取消</button>
                            <button
                                onClick={startLullaby}
                                style={{
                                    flex: 1, padding: "10px 0", borderRadius: 10, border: "none",
                                    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                                    color: "#fff", fontSize: "calc(14px*var(--app-text-scale,1))", fontWeight: 600, cursor: "pointer",
                                }}
                            >🌙 开始哄睡</button>
                        </div>
                    </div>
                </div>
            )}

        </div>
    );
}

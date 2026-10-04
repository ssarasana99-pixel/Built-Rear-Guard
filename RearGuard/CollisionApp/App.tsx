import React, {
  useState,
  useRef,
  useCallback,
  useEffect,
} from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Dimensions,
  StatusBar,
  Animated,
  Platform,
  ScrollView,
  Image,
  Alert,
} from 'react-native';
import {launchCamera, launchImageLibrary} from 'react-native-image-picker';
import {Camera, useCameraDevice} from 'react-native-vision-camera';

// ─── CONFIG ───────────────────────────────────────────────────────────────────
const SERVER_URL        = 'http://192.168.100.126:5001/process';
const VIDEO_SUMMARY_URL = 'http://192.168.100.126:5001/process_video_summary';

/**
 * Milliseconds between takePhoto() calls in live mode.
 * Lower = faster but more load. 400ms ≈ ~2-3 detections/sec.
 */
const LIVE_POLL_MS = 400;

const {width: SCREEN_W, height: SCREEN_H} = Dimensions.get('window');

// ─── Zone config ──────────────────────────────────────────────────────────────
const ZONE_CONFIG = {
  danger: {
    color: '#FF2D2D',
    bg: '#2D0000',
    label: 'DANGER',
    emoji: '🚨',
    advice: 'STOP IMMEDIATELY',
    speed: '0 km/h',
  },
  warning: {
    color: '#FF8C00',
    bg: '#2D1A00',
    label: 'WARNING',
    emoji: '⚠️',
    advice: 'SLOW DOWN',
    speed: '≤ 30 km/h',
  },
  safe: {
    color: '#00E676',
    bg: '#002D0F',
    label: 'SAFE',
    emoji: '✅',
    advice: 'CLEAR ROAD',
    speed: '≤ 60 km/h',
  },
  'no object': {
    color: '#90A4AE',
    bg: '#0D1117',
    label: 'SCANNING',
    emoji: '🔍',
    advice: 'No obstacle detected',
    speed: '—',
  },
} as const;

type ZoneKey = keyof typeof ZONE_CONFIG;

// ─── Types ────────────────────────────────────────────────────────────────────
interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface DetectedObject {
  label: string;
  distance: number;
  confidence?: number;
  bbox?: BBox;
}

interface DetectionResult {
  distance: number | null;
  zone: ZoneKey;
  objects: DetectedObject[];
}

interface VideoFrame extends DetectionResult {
  frame_index: number;
  timestamp: number;
}

interface VideoSummary {
  frames: VideoFrame[];
  worst_zone: ZoneKey;
  min_distance: number | null;
  total_frames_processed: number;
  video_duration: number;
}

type MediaMode = 'image' | 'video' | 'live';

// ─── Network helpers ──────────────────────────────────────────────────────────
async function sendImageToBackend(uri: string): Promise<DetectionResult> {
  const formData = new FormData();
  formData.append('image', {uri, type: 'image/jpeg', name: 'frame.jpg'} as unknown as Blob);
  const res = await fetch(SERVER_URL, {
    method: 'POST',
    body: formData,
    headers: {'Content-Type': 'multipart/form-data'},
  });
  if (!res.ok) {throw new Error(`Server error: ${res.status}`);}
  return res.json() as Promise<DetectionResult>;
}

async function sendVideoSummary(uri: string, frameStep = 5): Promise<VideoSummary> {
  const formData = new FormData();
  formData.append('video', {uri, type: 'video/mp4', name: 'clip.mp4'} as unknown as Blob);
  const res = await fetch(`${VIDEO_SUMMARY_URL}?frame_step=${frameStep}`, {
    method: 'POST',
    body: formData,
    headers: {'Content-Type': 'multipart/form-data'},
  });
  if (!res.ok) {throw new Error(`Server error: ${res.status}`);}
  return res.json() as Promise<VideoSummary>;
}

// ─── BBox overlay ─────────────────────────────────────────────────────────────
function BBoxOverlay({
  objects,
  zoneColor,
  containerW,
  containerH,
}: {
  objects: DetectedObject[];
  zoneColor: string;
  containerW: number;
  containerH: number;
}) {
  return (
    <>
      {objects.map((obj, i) => {
        if (!obj.bbox) {return null;}
        const {x1, y1, x2, y2} = obj.bbox;

        const left   = x1 * containerW;
        const top    = y1 * containerH;
        const width  = (x2 - x1) * containerW;
        const height = (y2 - y1) * containerH;

        if (width <= 0 || height <= 0) {return null;}

        return (
          <View
            key={i}
            style={[
              styles.bboxRect,
              {left, top, width, height, borderColor: zoneColor},
            ]}
            pointerEvents="none">
            <View style={[styles.bboxLabel, {backgroundColor: zoneColor + 'DD'}]}>
              <Text style={styles.bboxLabelText} numberOfLines={1}>
                {obj.label}  {obj.distance.toFixed(1)}m
                {obj.confidence != null
                  ? `  ${Math.round(obj.confidence * 100)}%`
                  : ''}
              </Text>
            </View>
          </View>
        );
      })}
    </>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [mediaMode, setMediaMode]         = useState<MediaMode>('image');
  const [result, setResult]               = useState<DetectionResult | null>(null);
  const [imageUri, setImageUri]           = useState<string | null>(null);
  const [videoUri, setVideoUri]           = useState<string | null>(null);
  const [isLoading, setIsLoading]         = useState(false);
  const [lastError, setLastError]         = useState<string | null>(null);

  // Video-specific state
  const [videoFrames, setVideoFrames]         = useState<VideoFrame[]>([]);
  const [currentFrameIdx, setCurrentFrameIdx] = useState(0);
  const [videoSummary, setVideoSummary]       = useState<VideoSummary | null>(null);
  const [isVideoPlaying, setIsVideoPlaying]   = useState(false);

  // Live-mode state
  const [liveActive, setLiveActive]               = useState(false);
  const [cameraPermission, setCameraPermission]   = useState<string | null>(null);
  const [liveFps, setLiveFps]                     = useState(0);
  const [liveFrameCount, setLiveFrameCount]       = useState(0);
  const [cameraLayout, setCameraLayout]           = useState({w: SCREEN_W, h: SCREEN_H});

  // Refs
  const cameraRef       = useRef<Camera>(null);
  const liveActiveRef   = useRef(false);
  const pollTimerRef    = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isSendingRef    = useRef(false);
  const fpsCountRef     = useRef(0);
  const fpsTimerRef     = useRef<ReturnType<typeof setInterval> | null>(null);
  const playIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fadeAnim  = useRef(new Animated.Value(0)).current;
  const pulseAnim = useRef(new Animated.Value(1)).current;

  // ── FIX 1: useCameraDevice('back') replaces useCameraDevices().back ──────────
  const device = useCameraDevice('back');

  // ── Camera permission ────────────────────────────────────────────────────────
  useEffect(() => {
    // FIX 2: typed 'status' parameter
    Camera.requestCameraPermission().then((status: string) => setCameraPermission(status));
  }, []);

  // ── Animations ───────────────────────────────────────────────────────────────
  const animateResult = useCallback((zone: ZoneKey) => {
    fadeAnim.setValue(0);
    Animated.timing(fadeAnim, {toValue: 1, duration: 250, useNativeDriver: true}).start();

    pulseAnim.stopAnimation();
    pulseAnim.setValue(1);
    if (zone === 'danger') {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {toValue: 1.04, duration: 280, useNativeDriver: true}),
          Animated.timing(pulseAnim, {toValue: 1.0,  duration: 280, useNativeDriver: true}),
        ]),
      ).start();
    }
  }, [fadeAnim, pulseAnim]);

  const handleResult = useCallback((data: DetectionResult) => {
    setResult(data);
    setLastError(null);
    animateResult(data.zone);
  }, [animateResult]);

  // ── Video playback ───────────────────────────────────────────────────────────
  const stopPlayback = useCallback(() => {
    if (playIntervalRef.current) {
      clearInterval(playIntervalRef.current);
      playIntervalRef.current = null;
    }
    setIsVideoPlaying(false);
  }, []);

  const startPlayback = useCallback(() => {
    if (videoFrames.length === 0) {return;}
    setIsVideoPlaying(true);
    setCurrentFrameIdx(0);
    playIntervalRef.current = setInterval(() => {
      setCurrentFrameIdx(prev => {
        const next = prev + 1;
        if (next >= videoFrames.length) {stopPlayback(); return prev;}
        return next;
      });
    }, 200);
  }, [videoFrames, stopPlayback]);

  useEffect(() => {
    if (mediaMode !== 'video' || videoFrames.length === 0) {return;}
    const frame = videoFrames[currentFrameIdx];
    if (frame) {handleResult(frame);}
  }, [currentFrameIdx, videoFrames, mediaMode, handleResult]);

  // ── Live FPS counter ─────────────────────────────────────────────────────────
  useEffect(() => {
    if (liveActive) {
      fpsTimerRef.current = setInterval(() => {
        setLiveFps(fpsCountRef.current);
        fpsCountRef.current = 0;
      }, 1000);
    } else {
      if (fpsTimerRef.current) {clearInterval(fpsTimerRef.current);}
      setLiveFps(0);
      fpsCountRef.current = 0;
    }
    return () => {if (fpsTimerRef.current) {clearInterval(fpsTimerRef.current);}};
  }, [liveActive]);

  // Cleanup on unmount
  useEffect(() => () => {
    stopPlayback();
    liveActiveRef.current = false;
    if (pollTimerRef.current) {clearTimeout(pollTimerRef.current);}
    if (fpsTimerRef.current) {clearInterval(fpsTimerRef.current);}
  }, [stopPlayback]);

  // ── takePhoto() polling loop ──────────────────────────────────────────────────
  const runPollLoop = useCallback(async () => {
    if (!liveActiveRef.current || !cameraRef.current) {return;}
    if (isSendingRef.current) {
      pollTimerRef.current = setTimeout(runPollLoop, 100);
      return;
    }

    isSendingRef.current = true;
    try {
      const photo = await cameraRef.current.takePhoto({
        flash: 'off',
      });

      if (!liveActiveRef.current) {isSendingRef.current = false; return;}

      const uri  = `file://${photo.path}`;
      const data = await sendImageToBackend(uri);

      if (liveActiveRef.current) {
        handleResult(data);
        fpsCountRef.current  += 1;
        setLiveFrameCount(c => c + 1);
      }
    } catch (e: unknown) {
      // FIX 3: typed unknown catch parameter
      if (liveActiveRef.current) {
        const msg = e instanceof Error ? e.message : 'Network / camera error';
        setLastError(msg);
      }
    } finally {
      isSendingRef.current = false;
      if (liveActiveRef.current) {
        pollTimerRef.current = setTimeout(runPollLoop, LIVE_POLL_MS);
      }
    }
  }, [handleResult]);

  // ── Start / Stop live ────────────────────────────────────────────────────────
  const startLive = useCallback(async () => {
    let perm = cameraPermission;
    if (perm !== 'authorized' && perm !== 'granted') {
      perm = await Camera.requestCameraPermission();
      setCameraPermission(perm);
    }
    if (perm !== 'authorized' && perm !== 'granted') {
      Alert.alert('Permission required', 'Camera access is needed for Live mode.');
      return;
    }
    setResult(null);
    setLastError(null);
    setLiveFrameCount(0);
    liveActiveRef.current = true;
    setLiveActive(true);

    pollTimerRef.current = setTimeout(runPollLoop, 600);
  }, [cameraPermission, runPollLoop]);

  const stopLive = useCallback(() => {
    liveActiveRef.current = false;
    isSendingRef.current  = false;
    if (pollTimerRef.current) {
      clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    setLiveActive(false);
    setResult(null);
  }, []);

  // ── Mode switching ───────────────────────────────────────────────────────────
  const switchMode = useCallback((mode: MediaMode) => {
    if (liveActive) {stopLive();}
    setMediaMode(mode);
    setResult(null);
    setLastError(null);
    if (mode !== 'image') {setImageUri(null);}
    if (mode !== 'video') {setVideoUri(null); setVideoFrames([]); setVideoSummary(null);}
  }, [liveActive, stopLive]);

  // ── Image processing ─────────────────────────────────────────────────────────
  const processImageUri = useCallback(async (uri: string) => {
    setImageUri(uri);
    setIsLoading(true);
    setResult(null);
    setLastError(null);
    try {
      const data = await sendImageToBackend(uri);
      handleResult(data);
    } catch (e: unknown) {
      // FIX 4: typed unknown catch parameter
      const msg = e instanceof Error ? e.message : 'Network error — check server IP';
      setLastError(msg);
    } finally {
      setIsLoading(false);
    }
  }, [handleResult]);

  // ── Video processing ─────────────────────────────────────────────────────────
  const processVideoUri = useCallback(async (uri: string) => {
    setVideoUri(uri);
    setIsLoading(true);
    setResult(null);
    setLastError(null);
    setVideoFrames([]);
    setVideoSummary(null);
    setCurrentFrameIdx(0);
    stopPlayback();
    try {
      const summary = await sendVideoSummary(uri, 5);
      const frames  = summary.frames as VideoFrame[];
      setVideoSummary(summary);
      setVideoFrames(frames);
      const worstIdx = frames.findIndex(f => f.zone === summary.worst_zone);
      const showIdx  = worstIdx >= 0 ? worstIdx : 0;
      setCurrentFrameIdx(showIdx);
      if (frames[showIdx]) {handleResult(frames[showIdx]);}
    } catch (e: unknown) {
      // FIX 5: typed unknown catch parameter
      const msg = e instanceof Error ? e.message : 'Network error — check server IP';
      setLastError(msg);
    } finally {
      setIsLoading(false);
    }
  }, [handleResult, stopPlayback]);

  // ── Launchers ─────────────────────────────────────────────────────────────────
  const captureImage = useCallback(() => {
    launchCamera(
      {mediaType: 'photo', quality: 0.7, cameraType: 'back', saveToPhotos: false},
      res => {if (!res.didCancel && res.assets) {processImageUri(res.assets[0].uri!);}}
    );
  }, [processImageUri]);

  const uploadImage = useCallback(() => {
    launchImageLibrary(
      {mediaType: 'photo', quality: 0.7, selectionLimit: 1},
      res => {if (!res.didCancel && res.assets) {processImageUri(res.assets[0].uri!);}}
    );
  }, [processImageUri]);

  const uploadVideo = useCallback(() => {
    launchImageLibrary(
      {mediaType: 'video', selectionLimit: 1, videoQuality: 'medium'},
      res => {if (!res.didCancel && res.assets) {processVideoUri(res.assets[0].uri!);}}
    );
  }, [processVideoUri]);

  const recordVideo = useCallback(() => {
    launchCamera(
      {mediaType: 'video', cameraType: 'back', videoQuality: 'medium', durationLimit: 30, saveToPhotos: false},
      res => {if (!res.didCancel && res.assets) {processVideoUri(res.assets[0].uri!);}}
    );
  }, [processVideoUri]);

  // ── Derived ───────────────────────────────────────────────────────────────────
  const zone    = result ? (ZONE_CONFIG[result.zone] ?? ZONE_CONFIG['no object']) : ZONE_CONFIG['no object'];
  const bgColor = mediaMode === 'live' ? '#000' : (result ? zone.bg : '#0D1117');

  const totalVideoFrames = videoFrames.length;
  const progressPct      = totalVideoFrames > 0
    ? Math.round((currentFrameIdx / Math.max(totalVideoFrames - 1, 1)) * 100)
    : 0;

  // ═══════════════════════════════════════════════════════════════════════════
  // LIVE MODE
  // ═══════════════════════════════════════════════════════════════════════════
  if (mediaMode === 'live') {
    return (
      <View style={styles.liveRoot}>
        <StatusBar barStyle="light-content" backgroundColor="#000" />

        {/* ── Camera viewfinder ── */}
        {device ? (
          <Camera
            ref={cameraRef}
            style={StyleSheet.absoluteFill}
            device={device}
            isActive={true}
            photo={true}
            video={false}
            audio={false}
            onLayout={e => {
              const {width: w, height: h} = e.nativeEvent.layout;
              setCameraLayout({w, h});
            }}
          />
        ) : (
          <View style={styles.livePlaceholder}>
            <Text style={styles.livePlaceholderIcon}>📷</Text>
            <Text style={styles.livePlaceholderText}>No back camera found</Text>
          </View>
        )}

        {/* ── Bounding boxes ── */}
        {result && liveActive && (
          <View style={StyleSheet.absoluteFill} pointerEvents="none">
            <BBoxOverlay
              objects={result.objects}
              zoneColor={zone.color}
              containerW={cameraLayout.w}
              containerH={cameraLayout.h}
            />
          </View>
        )}

        {/* ── Top bar ── */}
        <View style={styles.liveTopBar}>
          <TouchableOpacity style={styles.liveBackBtn} onPress={() => switchMode('image')}>
            <Text style={styles.liveBackText}>✕</Text>
          </TouchableOpacity>
          <Text style={styles.liveTitle}>REAR GUARD  LIVE</Text>
          {liveActive && (
            <View style={styles.fpsBadge}>
              <Text style={styles.fpsText}>{liveFps} fps</Text>
            </View>
          )}
        </View>

        {/* ── Zone pill ── */}
        {result && liveActive && (
          <Animated.View
            style={[
              styles.liveZonePill,
              {borderColor: zone.color, backgroundColor: zone.bg + 'DD'},
              {transform: [{scale: pulseAnim}]},
            ]}>
            <Text style={styles.liveZoneEmoji}>{zone.emoji}</Text>
            <View>
              <Text style={[styles.liveZoneLabel, {color: zone.color}]}>{zone.label}</Text>
              <Text style={[styles.liveZoneAdvice, {color: zone.color + 'CC'}]}>{zone.advice}</Text>
            </View>
          </Animated.View>
        )}

        {/* ── Objects list ── */}
        {result && liveActive && result.objects.length > 0 && (
          <View style={styles.liveObjectsBox}>
            <Text style={styles.liveObjectsTitle}>DETECTED</Text>
            {result.objects.map((obj, i) => (
              <View key={i} style={styles.liveObjectRow}>
                <View style={[styles.liveObjectDot, {backgroundColor: zone.color}]} />
                <Text style={styles.liveObjectLabel}>{obj.label}</Text>
                {obj.confidence != null && (
                  <Text style={styles.liveObjectConf}>{Math.round(obj.confidence * 100)}%</Text>
                )}
                <Text style={[styles.liveObjectDist, {color: zone.color}]}>
                  {obj.distance.toFixed(1)} m
                </Text>
              </View>
            ))}
          </View>
        )}

        {/* ── Metrics row ── */}
        {result && liveActive && (
          <View style={styles.liveMetricsRow}>
            <View style={[styles.liveMetricPill, {borderColor: zone.color}]}>
              <Text style={styles.liveMetricValue}>
                {result.distance != null ? result.distance.toFixed(1) : '—'}m
              </Text>
              <Text style={styles.liveMetricLabel}>distance</Text>
            </View>
            <View style={[styles.liveMetricPill, {borderColor: zone.color}]}>
              <Text style={[styles.liveMetricValue, {color: zone.color}]}>{zone.speed}</Text>
              <Text style={styles.liveMetricLabel}>max speed</Text>
            </View>
            <View style={[styles.liveMetricPill, {borderColor: '#546E7A'}]}>
              <Text style={styles.liveMetricValue}>{liveFrameCount}</Text>
              <Text style={styles.liveMetricLabel}>frames</Text>
            </View>
          </View>
        )}

        {/* ── Not-started placeholder ── */}
        {!liveActive && device && (
          <View style={styles.liveIdleBanner}>
            <Text style={styles.liveIdleText}>Point at the road and tap START</Text>
          </View>
        )}

        {/* ── Error ── */}
        {lastError && liveActive && (
          <View style={styles.liveErrorBox}>
            <Text style={styles.liveErrorText}>⚠  {lastError}</Text>
          </View>
        )}

        {/* ── Start / Stop ── */}
        <View style={styles.liveControlRow}>
          <TouchableOpacity
            style={[styles.liveStartBtn, {backgroundColor: liveActive ? '#FF2D2D' : '#00E676'}]}
            onPress={liveActive ? stopLive : startLive}
            activeOpacity={0.85}>
            <Text style={styles.liveStartBtnText}>
              {liveActive ? '⏹  STOP' : '▶  START'}
            </Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // NORMAL MODE (Photo / Video)
  // ═══════════════════════════════════════════════════════════════════════════
  return (
    <View style={[styles.root, {backgroundColor: bgColor}]}>
      <StatusBar barStyle="light-content" backgroundColor={bgColor} />
      <ScrollView
        contentContainerStyle={styles.scroll}
        showsVerticalScrollIndicator={false}>

        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.headerTitle}>REAR GUARD</Text>
          <Text style={styles.headerSub}>Collision Avoidance System</Text>
        </View>

        {/* 3-tab mode toggle */}
        <View style={styles.modeRow}>
          {(['image', 'video', 'live'] as const).map((m: MediaMode) => (
            <TouchableOpacity
              key={m}
              style={[styles.modeBtn, mediaMode === m && styles.modeBtnActive]}
              onPress={() => switchMode(m)}
              activeOpacity={0.8}>
              <Text
                style={[
                  styles.modeBtnText,
                  mediaMode === m && {color: String(m) === 'live' ? '#FF2D2D' : '#ECEFF1'},
                ]}>
                {m === 'image' ? '📷 Photo' : m === 'video' ? '🎥 Video' : '🔴 Live'}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* IMAGE buttons */}
        {mediaMode === 'image' && (
          <View style={styles.btnRow}>
            <TouchableOpacity
              style={[styles.actionBtn, {borderColor: '#00B0FF'}]}
              onPress={captureImage}
              activeOpacity={0.8}
              disabled={isLoading}>
              <Text style={styles.actionBtnIcon}>📷</Text>
              <Text style={[styles.actionBtnText, {color: '#00B0FF'}]}>Camera</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.actionBtn, {borderColor: '#AA00FF'}]}
              onPress={uploadImage}
              activeOpacity={0.8}
              disabled={isLoading}>
              <Text style={styles.actionBtnIcon}>🖼️</Text>
              <Text style={[styles.actionBtnText, {color: '#AA00FF'}]}>Upload</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* VIDEO buttons */}
        {mediaMode === 'video' && (
          <View style={styles.btnRow}>
            <TouchableOpacity
              style={[styles.actionBtn, {borderColor: '#FF4081'}]}
              onPress={recordVideo}
              activeOpacity={0.8}
              disabled={isLoading}>
              <Text style={styles.actionBtnIcon}>🔴</Text>
              <Text style={[styles.actionBtnText, {color: '#FF4081'}]}>Record</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.actionBtn, {borderColor: '#FF9100'}]}
              onPress={uploadVideo}
              activeOpacity={0.8}
              disabled={isLoading}>
              <Text style={styles.actionBtnIcon}>📂</Text>
              <Text style={[styles.actionBtnText, {color: '#FF9100'}]}>Pick Video</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Image preview */}
        {imageUri && mediaMode === 'image' && (
          <View style={styles.previewBox}>
            <Image source={{uri: imageUri}} style={styles.preview} resizeMode="cover" />
            {result && result.objects.length > 0 && !isLoading && (
              <View style={StyleSheet.absoluteFill} pointerEvents="none">
                <BBoxOverlay
                  objects={result.objects}
                  zoneColor={zone.color}
                  containerW={SCREEN_W - 44}
                  containerH={220}
                />
              </View>
            )}
            {isLoading && (
              <View style={styles.loadingOverlay}>
                <Text style={styles.loadingText}>⏳  Analyzing…</Text>
              </View>
            )}
          </View>
        )}

        {/* Video status card */}
        {videoUri && mediaMode === 'video' && (
          <View style={styles.videoStatusBox}>
            <Text style={styles.videoStatusIcon}>🎬</Text>
            <Text style={styles.videoStatusText}>
              {isLoading
                ? 'Analyzing video…'
                : videoSummary
                  ? `${videoSummary.total_frames_processed} frames  •  ${videoSummary.video_duration.toFixed(1)}s`
                  : 'Video ready'}
            </Text>

            {totalVideoFrames > 0 && !isLoading && (
              <View style={styles.scrubberWrap}>
                <View style={styles.scrubberTrack}>
                  {/* FIX 6: proper typing for percentage width */}
                  <View style={[styles.scrubberFill, {width: `${progressPct}%` as `${number}%`}]} />
                </View>
                <View style={styles.playControls}>
                  <TouchableOpacity
                    onPress={() => setCurrentFrameIdx(i => Math.max(0, i - 1))}
                    style={styles.playCtrlBtn}>
                    <Text style={styles.playCtrlIcon}>⏮</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    onPress={isVideoPlaying ? stopPlayback : startPlayback}
                    style={[styles.playCtrlBtn, styles.playCtrlBtnMain]}>
                    <Text style={styles.playCtrlIcon}>{isVideoPlaying ? '⏸' : '▶️'}</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    onPress={() => setCurrentFrameIdx(i => Math.min(totalVideoFrames - 1, i + 1))}
                    style={styles.playCtrlBtn}>
                    <Text style={styles.playCtrlIcon}>⏭</Text>
                  </TouchableOpacity>
                </View>
                <Text style={styles.frameCounter}>
                  Frame {currentFrameIdx + 1} / {totalVideoFrames}
                  {'  '}
                  {videoFrames[currentFrameIdx]
                    ? `(${videoFrames[currentFrameIdx].timestamp.toFixed(2)}s)`
                    : ''}
                </Text>
              </View>
            )}
          </View>
        )}

        {/* Video summary banner */}
        {videoSummary && !isLoading && mediaMode === 'video' && (
          <View style={[styles.summaryBanner, {borderColor: ZONE_CONFIG[videoSummary.worst_zone].color}]}>
            <Text style={styles.summaryTitle}>VIDEO SUMMARY</Text>
            <View style={styles.summaryRow}>
              <View style={styles.summaryItem}>
                <Text style={[styles.summaryValue, {color: ZONE_CONFIG[videoSummary.worst_zone].color}]}>
                  {ZONE_CONFIG[videoSummary.worst_zone].emoji}  {videoSummary.worst_zone.toUpperCase()}
                </Text>
                <Text style={styles.summaryLabel}>worst zone</Text>
              </View>
              <View style={styles.summaryItem}>
                <Text style={styles.summaryValue}>
                  {videoSummary.min_distance != null
                    ? videoSummary.min_distance.toFixed(2)
                    : '—'}m
                </Text>
                <Text style={styles.summaryLabel}>closest object</Text>
              </View>
            </View>
          </View>
        )}

        {/* Error */}
        {lastError && (
          <View style={styles.errorBox}>
            <Text style={styles.errorText}>⚠  {lastError}</Text>
          </View>
        )}

        {/* Detection results */}
        {result && !isLoading && (
          <Animated.View style={{opacity: fadeAnim, width: '100%'}}>
            <Animated.View
              style={[
                styles.zoneCard,
                {borderColor: zone.color, transform: [{scale: pulseAnim}]},
              ]}>
              <Text style={styles.zoneEmoji}>{zone.emoji}</Text>
              <Text style={[styles.zoneLabel, {color: zone.color}]}>{zone.label}</Text>
              <Text style={[styles.zoneAdvice, {color: zone.color}]}>{zone.advice}</Text>
            </Animated.View>

            <View style={styles.metricsRow}>
              <View style={styles.metricBox}>
                <Text style={styles.metricValue}>
                  {result.distance != null ? result.distance.toFixed(2) : '—'}
                </Text>
                <Text style={styles.metricLabel}>metres away</Text>
              </View>
              <View style={styles.metricBox}>
                <Text style={[styles.metricValue, {color: zone.color, fontSize: 22}]}>
                  {zone.speed}
                </Text>
                <Text style={styles.metricLabel}>recommended speed</Text>
              </View>
            </View>

            <View style={styles.objectsBox}>
              <Text style={styles.objectsTitle}>DETECTED OBJECTS</Text>
              {result.objects.length > 0 ? (
                result.objects.map((obj, i) => (
                  <View key={i} style={styles.objectRow}>
                    <View style={[styles.objectDot, {backgroundColor: zone.color}]} />
                    <Text style={styles.objectLabel}>{obj.label}</Text>
                    {obj.confidence != null && (
                      <Text style={styles.objectConf}>
                        {Math.round(obj.confidence * 100)}%
                      </Text>
                    )}
                    <Text style={[styles.objectDist, {color: zone.color}]}>
                      {obj.distance.toFixed(2)} m
                    </Text>
                  </View>
                ))
              ) : (
                <Text style={styles.noObjects}>
                  No relevant obstacles found — try a clearer image with a car, person, or truck
                </Text>
              )}
            </View>
          </Animated.View>
        )}

        {/* Placeholder */}
        {!imageUri && !videoUri && (
          <View style={styles.placeholder}>
            <Text style={styles.placeholderIcon}>
              {mediaMode === 'video' ? '🎥' : '🚗'}
            </Text>
            <Text style={styles.placeholderText}>
              {mediaMode === 'video'
                ? 'Record a clip or pick a video\nfrom your gallery'
                : 'Tap Camera to take a photo\nor Upload to pick from gallery'}
            </Text>
          </View>
        )}
      </ScrollView>
    </View>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
  // ── Normal mode ──────────────────────────────────────────────────────────────
  root: {flex: 1},
  scroll: {
    paddingHorizontal: 22,
    paddingTop: Platform.OS === 'ios' ? 60 : 40,
    paddingBottom: 40,
    alignItems: 'center',
  },

  header: {alignItems: 'center', marginBottom: 20},
  headerTitle: {fontSize: 28, fontWeight: '900', color: '#ECEFF1', letterSpacing: 6},
  headerSub: {fontSize: 11, color: '#546E7A', letterSpacing: 2, marginTop: 4},

  modeRow: {
    flexDirection: 'row',
    width: '100%',
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 14,
    padding: 4,
    marginBottom: 16,
  },
  modeBtn: {flex: 1, paddingVertical: 10, alignItems: 'center', borderRadius: 11},
  modeBtnActive: {backgroundColor: 'rgba(255,255,255,0.12)'},
  modeBtnText: {fontSize: 12, fontWeight: '700', color: '#546E7A', letterSpacing: 0.5},

  btnRow: {
    flexDirection: 'row',
    width: '100%',
    justifyContent: 'space-between',
    marginBottom: 20,
    gap: 14,
  },
  actionBtn: {
    flex: 1,
    borderWidth: 1.5,
    borderRadius: 16,
    paddingVertical: 20,
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.04)',
  },
  actionBtnIcon: {fontSize: 28, marginBottom: 6},
  actionBtnText: {fontSize: 13, fontWeight: '700', letterSpacing: 1},

  previewBox: {
    width: '100%',
    height: 220,
    borderRadius: 16,
    overflow: 'hidden',
    marginBottom: 16,
    backgroundColor: '#1A1A2E',
  },
  preview: {width: '100%', height: '100%'},
  loadingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.65)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {color: '#ECEFF1', fontSize: 16, fontWeight: '700'},

  videoStatusBox: {
    width: '100%',
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 16,
    padding: 18,
    marginBottom: 16,
    alignItems: 'center',
  },
  videoStatusIcon: {fontSize: 32, marginBottom: 8},
  videoStatusText: {color: '#ECEFF1', fontSize: 13, fontWeight: '600', marginBottom: 12},

  scrubberWrap: {width: '100%', alignItems: 'center'},
  scrubberTrack: {
    width: '100%',
    height: 4,
    backgroundColor: 'rgba(255,255,255,0.1)',
    borderRadius: 2,
    marginBottom: 12,
  },
  scrubberFill: {height: '100%', backgroundColor: '#00B0FF', borderRadius: 2},
  playControls: {flexDirection: 'row', gap: 16, alignItems: 'center', marginBottom: 8},
  playCtrlBtn: {padding: 10, borderRadius: 12, backgroundColor: 'rgba(255,255,255,0.08)'},
  playCtrlBtnMain: {paddingHorizontal: 20, backgroundColor: 'rgba(0,176,255,0.2)'},
  playCtrlIcon: {fontSize: 20},
  frameCounter: {color: '#546E7A', fontSize: 11, letterSpacing: 1},

  summaryBanner: {
    width: '100%',
    borderWidth: 1.5,
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
    backgroundColor: 'rgba(255,255,255,0.04)',
  },
  summaryTitle: {color: '#546E7A', fontSize: 10, letterSpacing: 2, marginBottom: 12},
  summaryRow: {flexDirection: 'row', justifyContent: 'space-around'},
  summaryItem: {alignItems: 'center'},
  summaryValue: {fontSize: 20, fontWeight: '900', color: '#ECEFF1'},
  summaryLabel: {fontSize: 10, color: '#546E7A', marginTop: 4, letterSpacing: 1},

  errorBox: {
    width: '100%',
    backgroundColor: '#2D0000',
    borderRadius: 12,
    padding: 14,
    marginBottom: 16,
  },
  errorText: {color: '#FF5252', fontSize: 13},

  zoneCard: {
    width: '100%',
    borderWidth: 2,
    borderRadius: 20,
    alignItems: 'center',
    paddingVertical: 26,
    marginBottom: 16,
    backgroundColor: 'rgba(255,255,255,0.04)',
  },
  zoneEmoji: {fontSize: 44, marginBottom: 8},
  zoneLabel: {fontSize: 30, fontWeight: '900', letterSpacing: 4},
  zoneAdvice: {fontSize: 13, marginTop: 6, letterSpacing: 1, opacity: 0.85},

  metricsRow: {flexDirection: 'row', width: '100%', gap: 14, marginBottom: 16},
  metricBox: {
    flex: 1,
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 14,
    alignItems: 'center',
    paddingVertical: 18,
  },
  metricValue: {fontSize: 36, fontWeight: '900', color: '#ECEFF1'},
  metricLabel: {fontSize: 11, color: '#546E7A', marginTop: 4, letterSpacing: 1},

  objectsBox: {
    width: '100%',
    backgroundColor: 'rgba(255,255,255,0.05)',
    borderRadius: 14,
    padding: 16,
    marginBottom: 16,
  },
  objectsTitle: {color: '#546E7A', fontSize: 10, letterSpacing: 2, marginBottom: 12},
  objectRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,255,255,0.05)',
    gap: 8,
  },
  objectDot: {width: 6, height: 6, borderRadius: 3},
  objectLabel: {flex: 1, color: '#ECEFF1', fontSize: 14, textTransform: 'capitalize'},
  objectConf: {color: '#546E7A', fontSize: 12},
  objectDist: {fontSize: 14, fontWeight: '700'},
  noObjects: {color: '#546E7A', fontSize: 13, fontStyle: 'italic', lineHeight: 20},

  placeholder: {alignItems: 'center', marginTop: 60},
  placeholderIcon: {fontSize: 60, marginBottom: 16},
  placeholderText: {color: '#37474F', fontSize: 14, textAlign: 'center', lineHeight: 22},

  // ── Bounding boxes ────────────────────────────────────────────────────────────
  bboxRect: {
    position: 'absolute',
    borderWidth: 2,
    borderRadius: 4,
  },
  bboxLabel: {
    position: 'absolute',
    top: -22,
    left: -1,
    borderRadius: 4,
    paddingHorizontal: 6,
    paddingVertical: 2,
    maxWidth: 200,
  },
  bboxLabelText: {
    color: '#000',
    fontSize: 11,
    fontWeight: '800',
    letterSpacing: 0.3,
  },

  // ── Live mode ─────────────────────────────────────────────────────────────────
  liveRoot: {flex: 1, backgroundColor: '#000'},

  livePlaceholder: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0D1117',
  },
  livePlaceholderIcon: {fontSize: 64, marginBottom: 16},
  livePlaceholderText: {color: '#546E7A', fontSize: 15, textAlign: 'center', lineHeight: 24},

  liveTopBar: {
    position: 'absolute',
    top: Platform.OS === 'ios' ? 56 : 36,
    left: 0,
    right: 0,
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 18,
    zIndex: 10,
  },
  liveBackBtn: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: 'rgba(0,0,0,0.55)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  liveBackText: {color: '#ECEFF1', fontSize: 16, fontWeight: '700'},
  liveTitle: {
    flex: 1,
    textAlign: 'center',
    color: '#ECEFF1',
    fontSize: 12,
    fontWeight: '900',
    letterSpacing: 3,
  },
  fpsBadge: {
    backgroundColor: 'rgba(0,0,0,0.55)',
    borderRadius: 10,
    paddingHorizontal: 10,
    paddingVertical: 4,
  },
  fpsText: {color: '#00E676', fontSize: 11, fontWeight: '700'},

  liveZonePill: {
    position: 'absolute',
    top: Platform.OS === 'ios' ? 110 : 90,
    left: 18,
    right: 18,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    borderWidth: 1.5,
    borderRadius: 18,
    paddingVertical: 14,
    paddingHorizontal: 20,
    zIndex: 10,
  },
  liveZoneEmoji: {fontSize: 32},
  liveZoneLabel: {fontSize: 22, fontWeight: '900', letterSpacing: 3},
  liveZoneAdvice: {fontSize: 11, letterSpacing: 1, marginTop: 2},

  liveObjectsBox: {
    position: 'absolute',
    bottom: 290,
    left: 18,
    right: 18,
    backgroundColor: 'rgba(0,0,0,0.72)',
    borderRadius: 14,
    padding: 12,
    zIndex: 10,
  },
  liveObjectsTitle: {
    color: '#546E7A',
    fontSize: 9,
    letterSpacing: 2,
    marginBottom: 8,
    fontWeight: '700',
  },
  liveObjectRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 5,
    gap: 8,
  },
  liveObjectDot: {width: 6, height: 6, borderRadius: 3},
  liveObjectLabel: {flex: 1, color: '#ECEFF1', fontSize: 13, textTransform: 'capitalize'},
  liveObjectConf: {color: '#546E7A', fontSize: 11},
  liveObjectDist: {fontSize: 13, fontWeight: '700'},

  liveMetricsRow: {
    position: 'absolute',
    bottom: 170,
    left: 18,
    right: 18,
    flexDirection: 'row',
    gap: 10,
    zIndex: 10,
  },
  liveMetricPill: {
    flex: 1,
    borderWidth: 1.5,
    borderRadius: 14,
    alignItems: 'center',
    paddingVertical: 12,
    backgroundColor: 'rgba(0,0,0,0.65)',
  },
  liveMetricValue: {fontSize: 22, fontWeight: '900', color: '#ECEFF1'},
  liveMetricLabel: {fontSize: 9, color: '#90A4AE', marginTop: 2, letterSpacing: 1},

  liveIdleBanner: {
    position: 'absolute',
    bottom: 170,
    left: 18,
    right: 18,
    backgroundColor: 'rgba(0,0,0,0.6)',
    borderRadius: 14,
    padding: 16,
    alignItems: 'center',
    zIndex: 10,
  },
  liveIdleText: {color: '#90A4AE', fontSize: 14, letterSpacing: 0.5},

  liveErrorBox: {
    position: 'absolute',
    top: Platform.OS === 'ios' ? 200 : 180,
    left: 18,
    right: 18,
    backgroundColor: 'rgba(45,0,0,0.9)',
    borderRadius: 10,
    padding: 10,
    zIndex: 10,
  },
  liveErrorText: {color: '#FF5252', fontSize: 12},

  liveControlRow: {
    position: 'absolute',
    bottom: Platform.OS === 'ios' ? 52 : 30,
    left: 0,
    right: 0,
    alignItems: 'center',
    zIndex: 10,
  },
  liveStartBtn: {
    width: 180,
    paddingVertical: 18,
    borderRadius: 40,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.5,
    shadowRadius: 10,
    elevation: 8,
  },
  liveStartBtnText: {color: '#000', fontSize: 18, fontWeight: '900', letterSpacing: 2},
});
"use client";

import { useRef, useCallback, useEffect, useState } from "react";
import { WS_BASE } from "@/lib/constants";
import type { Landmark } from "@/lib/types";

export interface WSResult {
  t: number;
  frame: number;
  landmarks: Landmark[] | null;
  mask: string | null; // base64 PNG
  bg_capture?: boolean;
  bg_frames_captured?: number;
}

export interface WSEvent {
  event: string;
  success?: boolean;
  has_background?: boolean;
  segmenter?: string;
  gpu?: { device: string; backend: string | null };
}

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [latestResult, setLatestResult] = useState<WSResult | null>(null);
  const [latestEvent, setLatestEvent] = useState<WSEvent | null>(null);

  const connect = useCallback((segmenter: string = "mediapipe") => {
    const ws = new WebSocket(`${WS_BASE}/ws/gameplay?segmenter=${segmenter}`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.event) {
        setLatestEvent(data as WSEvent);
      } else {
        setLatestResult(data as WSResult);
      }
    };

    return ws;
  }, []);

  const disconnect = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setConnected(false);
  }, []);

  const sendFrame = useCallback((blob: Blob) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      blob.arrayBuffer().then((buf) => ws.send(buf));
    }
  }, []);

  const sendCommand = useCallback((cmd: string) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ cmd }));
    }
  }, []);

  useEffect(() => {
    return () => {
      wsRef.current?.close();
    };
  }, []);

  return { connect, disconnect, sendFrame, sendCommand, connected, latestResult, latestEvent };
}

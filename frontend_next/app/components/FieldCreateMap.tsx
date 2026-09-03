"use client";

import {
  type Dispatch,
  type SetStateAction,
  useEffect,
  useRef,
  useState,
} from "react";
import { LocateFixed } from "lucide-react";

type LeafletModule = typeof import("leaflet");

export function FieldCreateMap({
  points,
  setPoints,
}: {
  points: number[][];
  setPoints: Dispatch<SetStateAction<number[][]>>;
}) {
  const container = useRef<HTMLDivElement>(null);
  const leaflet = useRef<LeafletModule | null>(null);
  const map = useRef<import("leaflet").Map | null>(null);
  const drawingLayer = useRef<import("leaflet").LayerGroup | null>(null);
  const locationLayer = useRef<import("leaflet").LayerGroup | null>(null);
  const [ready, setReady] = useState(false);
  const [locating, setLocating] = useState(false);
  const [message, setMessage] = useState("点击地图，依次标记地块边界点");

  useEffect(() => {
    let active = true;

    async function initializeMap() {
      const node = container.current;
      if (!node) return;
      const L = await import("leaflet");
      if (!active) return;

      const instance = L.map(node, {
        zoomControl: true,
        attributionControl: true,
        worldCopyJump: true,
      }).setView([39.9, 116.4], 12);

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap contributors",
      }).addTo(instance);

      const drawGroup = L.layerGroup().addTo(instance);
      const locateGroup = L.layerGroup().addTo(instance);
      instance.on("click", (event) => {
        setPoints((current) => [
          ...current,
          [
            Number(event.latlng.lng.toFixed(6)),
            Number(event.latlng.lat.toFixed(6)),
          ],
        ]);
      });

      leaflet.current = L;
      map.current = instance;
      drawingLayer.current = drawGroup;
      locationLayer.current = locateGroup;
      setReady(true);
      requestAnimationFrame(() => instance.invalidateSize());
    }

    initializeMap().catch(() => {
      if (active) setMessage("地图加载失败，请检查网络后重试");
    });

    return () => {
      active = false;
      map.current?.remove();
      map.current = null;
      leaflet.current = null;
      drawingLayer.current = null;
      locationLayer.current = null;
    };
  }, [setPoints]);

  useEffect(() => {
    const L = leaflet.current;
    const instance = map.current;
    const layer = drawingLayer.current;
    if (!ready || !L || !instance || !layer) return;

    layer.clearLayers();
    const latLngs = points.map(
      (point) => [Number(point[1]), Number(point[0])] as [number, number],
    );

    if (latLngs.length >= 3) {
      L.polygon(latLngs, {
        color: "#2e765d",
        weight: 3,
        fillColor: "#72a785",
        fillOpacity: 0.28,
      }).addTo(layer);
    } else if (latLngs.length >= 2) {
      L.polyline(latLngs, { color: "#2e765d", weight: 3 }).addTo(layer);
    }

    latLngs.forEach((point, index) => {
      L.circleMarker(point, {
        radius: 6,
        color: "#fff",
        weight: 2,
        fillColor: "#d59039",
        fillOpacity: 1,
      })
        .bindTooltip(`边界点 ${index + 1}`, { permanent: false })
        .addTo(layer);
    });
  }, [points, ready]);

  function locateDevice() {
    const L = leaflet.current;
    const instance = map.current;
    const layer = locationLayer.current;
    if (!L || !instance || !layer) return;
    if (!navigator.geolocation) {
      setMessage("当前浏览器不支持定位功能");
      return;
    }

    setLocating(true);
    setMessage("正在获取本机位置…");
    navigator.geolocation.getCurrentPosition(
      (position) => {
        const current: [number, number] = [
          position.coords.latitude,
          position.coords.longitude,
        ];
        layer.clearLayers();
        L.circle(current, {
          radius: Math.max(position.coords.accuracy, 10),
          color: "#3978b8",
          weight: 1,
          fillColor: "#6ba9e8",
          fillOpacity: 0.12,
        }).addTo(layer);
        L.circleMarker(current, {
          radius: 7,
          color: "#fff",
          weight: 3,
          fillColor: "#3978b8",
          fillOpacity: 1,
        })
          .bindTooltip("本机位置")
          .addTo(layer);
        instance.setView(current, 17, { animate: true });
        setMessage("已切换到本机位置，可继续点击地图标记边界");
        setLocating(false);
      },
      (error) => {
        const reason =
          error.code === error.PERMISSION_DENIED
            ? "定位权限未开启，请允许浏览器访问位置"
            : "暂时无法获取本机位置，请稍后重试";
        setMessage(reason);
        setLocating(false);
      },
      { enableHighAccuracy: true, timeout: 12000, maximumAge: 30000 },
    );
  }

  return (
    <div className="field-create-map-shell">
      <div className="draw-toolbar">
        <span>{message}</span>
        <div className="draw-toolbar-actions">
          <button
            type="button"
            className="secondary-button compact-button"
            onClick={locateDevice}
            disabled={!ready || locating}
          >
            <LocateFixed />
            {locating ? "定位中" : "定位到本机"}
          </button>
          <button
            type="button"
            className="secondary-button compact-button"
            onClick={() => setPoints([])}
            disabled={!points.length}
          >
            清除边界
          </button>
        </div>
      </div>
      <div ref={container} className="field-create-map" />
    </div>
  );
}

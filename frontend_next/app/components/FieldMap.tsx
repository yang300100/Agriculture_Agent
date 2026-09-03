"use client";

import { useEffect, useRef, useState } from "react";
import type { Field } from "../types";

function validCoordinate(point: number[]) {
  return (
    point.length >= 2 &&
    Number.isFinite(Number(point[0])) &&
    Number.isFinite(Number(point[1])) &&
    Math.abs(Number(point[0])) <= 180 &&
    Math.abs(Number(point[1])) <= 90
  );
}

function validCenter(field: Field) {
  return (
    Number.isFinite(Number(field.center_lat)) &&
    Number.isFinite(Number(field.center_lon)) &&
    Math.abs(Number(field.center_lat)) <= 90 &&
    Math.abs(Number(field.center_lon)) <= 180 &&
    !(Number(field.center_lat) === 0 && Number(field.center_lon) === 0)
  );
}

export function FieldOverviewMap({
  fields,
  selectedId,
  onSelect,
}: {
  fields: Field[];
  selectedId?: string;
  onSelect: (field: Field) => void;
}) {
  const container = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    let map: import("leaflet").Map | null = null;

    async function renderMap() {
      const node = container.current;
      if (!node) return;
      setLoading(true);
      setError("");

      const L = await import("leaflet");
      if (!active) return;

      const selected = fields.find(
        (field) => field.id === selectedId && validCenter(field),
      );
      const centerSource = selected || fields.find(validCenter);
      const center: [number, number] = centerSource
        ? [Number(centerSource.center_lat), Number(centerSource.center_lon)]
        : [39.9, 116.4];

      map = L.map(node, {
        zoomControl: true,
        attributionControl: true,
        worldCopyJump: true,
      }).setView(center, centerSource ? 14 : 5);

      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap contributors",
      }).addTo(map);

      const bounds = L.latLngBounds([]);
      fields.forEach((field) => {
        const coordinates = (field.coordinates || [])
          .filter(validCoordinate)
          .map(
            (point) => [Number(point[1]), Number(point[0])] as [number, number],
          );
        const isSelected = field.id === selectedId;
        const color = isSelected ? "#d59039" : "#2e765d";

        if (coordinates.length >= 3) {
          const polygon = L.polygon(coordinates, {
            color,
            weight: isSelected ? 4 : 2,
            fillColor: color,
            fillOpacity: isSelected ? 0.32 : 0.2,
          }).addTo(map!);
          polygon.bindTooltip(
            `${field.name} · ${field.current_crop || "未种植"} · ${Number(field.area_mu || 0).toFixed(2)} 亩`,
            { sticky: true },
          );
          polygon.on("click", () => onSelect(field));
          coordinates.forEach((point) => bounds.extend(point));
          return;
        }

        if (validCenter(field)) {
          const point: [number, number] = [
            Number(field.center_lat),
            Number(field.center_lon),
          ];
          const marker = L.circleMarker(point, {
            radius: isSelected ? 10 : 8,
            color: "#fff",
            weight: 3,
            fillColor: color,
            fillOpacity: 1,
          }).addTo(map!);
          marker.bindTooltip(`${field.name} · ${field.current_crop || "未种植"}`);
          marker.on("click", () => onSelect(field));
          bounds.extend(point);
        }
      });

      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [32, 32], maxZoom: 16 });
      }
      requestAnimationFrame(() => map?.invalidateSize());
      setLoading(false);
    }

    renderMap().catch((reason) => {
      if (!active) return;
      setLoading(false);
      setError(reason instanceof Error ? reason.message : "地图加载失败");
    });

    return () => {
      active = false;
      map?.remove();
    };
  }, [fields, onSelect, selectedId]);

  return (
    <div className="farm-map real-field-map">
      <div ref={container} className="leaflet-field-map" />
      {loading && <div className="map-status">正在加载地块地图…</div>}
      {error && <div className="map-status error">地图加载失败：{error}</div>}
      {!loading && !error && !fields.length && (
        <div className="map-status map-empty-state">
          地图已就绪，添加地块后会在这里显示边界。
        </div>
      )}
    </div>
  );
}

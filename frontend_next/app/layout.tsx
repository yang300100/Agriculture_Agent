import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

const baseMetadata: Metadata = {
  title: {
    default: "青禾智能农场",
    template: "%s · 青禾智能农场",
  },
  description:
    "覆盖种植、地块、农事、财务、设备与自动化的 AI 农业全周期运营平台。",
  applicationName: "青禾智能农场",
  keywords: ["智慧农业", "农场管理", "农业物联网", "AI 农业助手"],
};

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const host =
    requestHeaders.get("x-forwarded-host") ??
    requestHeaders.get("host") ??
    "localhost:3000";
  const protocol =
    requestHeaders.get("x-forwarded-proto") ??
    (host.startsWith("localhost") ? "http" : "https");
  const imageUrl = `${protocol}://${host}/og.png`;

  return {
    ...baseMetadata,
    openGraph: {
      title: "青禾智能农场",
      description: "AI 驱动的全周期农业运营台",
      type: "website",
      locale: "zh_CN",
      images: [
        { url: imageUrl, width: 1733, height: 908, alt: "青禾智能农场" },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title: "青禾智能农场",
      description: "AI 驱动的全周期农业运营台",
      images: [imageUrl],
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}

'use client';

import { useSidebar } from 'fumadocs-ui/layouts/docs/slots/sidebar';
import { useEffect, useRef } from 'react';

const STORAGE_KEY = 'wt-sidebar-width';
const DEFAULT_WIDTH = 268;
// Header row (search + collapse button + title) overflows below ~240px.
const MIN_WIDTH = 240;
const MAX_WIDTH = 480;
// Drag further left than this snaps the sidebar closed.
const COLLAPSE_THRESHOLD = 180;

// Fumadocs sets `--fd-sidebar-width` on several elements (grid container,
// sidebar placeholder, and the aside itself) via Tailwind arbitrary classes.
// Overriding via inherited CSS doesn't work because each of those declarations
// targets the element directly. So we write the inline var onto every element
// whose className mentions `--fd-sidebar-width`, with !important to beat the
// original class declaration.
function forEachTarget(fn: (el: HTMLElement) => void) {
  document.querySelectorAll<HTMLElement>('[class*="--fd-sidebar-width"]').forEach(fn);
}

function applyWidth(px: number) {
  if (typeof window === 'undefined' || window.innerWidth < 768) return;
  const value = `${px}px`;
  forEachTarget((el) => el.style.setProperty('--fd-sidebar-width', value, 'important'));
}

function clearWidth() {
  forEachTarget((el) => el.style.removeProperty('--fd-sidebar-width'));
}

export function SidebarResizer() {
  const { setCollapsed } = useSidebar();
  const draggingRef = useRef(false);

  useEffect(() => {
    const stored = Number(localStorage.getItem(STORAGE_KEY));
    const width = stored >= MIN_WIDTH && stored <= MAX_WIDTH ? stored : DEFAULT_WIDTH;
    // Reapply on viewport change; below md the drawer takes over, so strip
    // our inline var and let the layout's 0px class reclaim the grid column.
    const sync = () => {
      if (window.innerWidth < 768) clearWidth();
      else applyWidth(width);
    };
    sync();
    window.addEventListener('resize', sync);
    return () => window.removeEventListener('resize', sync);
  }, []);

  const onMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    draggingRef.current = true;
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    const onMove = (ev: MouseEvent) => {
      if (!draggingRef.current) return;
      const next = ev.clientX;
      if (next < COLLAPSE_THRESHOLD) {
        stopDrag();
        setCollapsed(true);
        return;
      }
      const clamped = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, next));
      applyWidth(clamped);
      localStorage.setItem(STORAGE_KEY, String(clamped));
    };

    const stopDrag = () => {
      draggingRef.current = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', stopDrag);
    };

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mouseup', stopDrag);
  };

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label="Resize sidebar"
      onMouseDown={onMouseDown}
      className="absolute inset-y-0 inset-e-0 w-1 z-30 cursor-col-resize hover:bg-fd-primary/30 active:bg-fd-primary/50 transition-colors max-md:hidden"
    />
  );
}

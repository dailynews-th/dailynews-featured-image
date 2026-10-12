// Service worker: the app works offline and opens fast after the first visit.
// The page registers sw.js?v=<version>, so every release installs a fresh cache and drops the old one.
const VERSION = new URL(self.location).searchParams.get("v") || "dev";
const CACHE = "dn-cover-" + VERSION;
const SHELL = ["./", "index.html", "manifest.webmanifest", "fonts/fonts.css",
  "fonts/Kanit-ExtraLight.ttf", "fonts/Kanit-Regular.ttf", "fonts/Kanit-Medium.ttf", "fonts/Kanit-SemiBold.ttf",
  "fonts/IBMPlexSansThai-Regular.ttf", "fonts/IBMPlexSansThai-Medium.ttf", "fonts/IBMPlexSansThai-SemiBold.ttf",
  "overlay.png", "web_overlay.png", "web_line.png", "logo_white.png", "logo_pink.png", "line_white.png", "line_pink.png",
  "icons/icon-192.png", "icons/icon-512.png"];
self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => Promise.all(SHELL.map(u => c.add(u).catch(() => {})))).then(() => self.skipWaiting()));
});
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k.startsWith("dn-cover-") && k !== CACHE && k !== "dn-cover-models").map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener("fetch", e => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== self.location.origin) return;
  // the page and the manifest (app name / icons): network first so changes show up right away, the cached copy when offline
  if (req.mode === "navigate" || url.pathname.endsWith("/index.html") || url.pathname.endsWith("/manifest.webmanifest")) {
    e.respondWith(fetch(req).then(r => { const c = r.clone(); caches.open(CACHE).then(k => k.put(req, c)); return r; })
      .catch(() => caches.match(req).then(r => r || caches.match("index.html"))));
    return;
  }
  // models (large, change rarely): kept in their own cache across releases
  const store = url.pathname.includes("/model/") ? "dn-cover-models" : CACHE;
  e.respondWith(caches.open(store).then(c => c.match(req).then(hit => hit || fetch(req).then(r => { if (r.ok) c.put(req, r.clone()); return r; }))));
});

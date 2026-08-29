/* Satellite — working log. No dependencies. */
(() => {
  "use strict";

  /* ── scrollspy ──────────────────────────────────────────────────── */
  const tabs = [...document.querySelectorAll(".tabs a")];
  const targets = tabs
    .map((a) => document.querySelector(a.getAttribute("href")))
    .filter(Boolean);

  if ("IntersectionObserver" in window && targets.length) {
    const seen = new Map();
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => seen.set(e.target.id, e.intersectionRatio));
        let best = null;
        seen.forEach((ratio, id) => {
          if (ratio > 0 && (!best || ratio > best.ratio)) best = { id, ratio };
        });
        tabs.forEach((a) =>
          a.classList.toggle("active", !!best && a.getAttribute("href") === "#" + best.id)
        );
      },
      { rootMargin: "-20% 0px -55% 0px", threshold: [0, 0.25, 0.5, 1] }
    );
    targets.forEach((t) => io.observe(t));
  }

  /* ── lightbox ───────────────────────────────────────────────────── */
  const lb = document.getElementById("lightbox");
  const lbImg = lb.querySelector("img");
  const lbCap = lb.querySelector(".lb-cap");

  const openLightbox = (img) => {
    lbImg.src = img.currentSrc || img.src;
    lbImg.alt = img.alt;
    const cap = img.closest("figure")?.querySelector("figcaption");
    lbCap.textContent = cap ? cap.textContent.trim() : img.alt;
    lb.hidden = false;
    document.body.style.overflow = "hidden";
  };
  const closeLightbox = () => {
    lb.hidden = true;
    lbImg.removeAttribute("src");
    document.body.style.overflow = "";
  };

  document.querySelectorAll(".fig img:not(.player img)").forEach((img) => {
    img.addEventListener("click", () => openLightbox(img));
  });
  lb.addEventListener("click", closeLightbox);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !lb.hidden) closeLightbox();
  });

  /* ── before / after slider ──────────────────────────────────────── */
  document.querySelectorAll(".compare").forEach((cmp) => {
    const labelBefore = cmp.dataset.labelBefore || "Before";
    const labelAfter = cmp.dataset.labelAfter || "After";
    const frame = document.createElement("div");
    frame.className = "compare-frame";
    frame.innerHTML = `
      <div class="layer layer-after"><img src="${cmp.dataset.after}" alt="${labelAfter}" loading="lazy"></div>
      <div class="layer layer-clip"><img src="${cmp.dataset.before}" alt="${labelBefore}" loading="lazy"></div>
      <span class="compare-tag left">${labelBefore}</span>
      <span class="compare-tag right">${labelAfter}</span>
      <div class="compare-handle"></div>`;
    cmp.prepend(frame);

    const clipped = frame.querySelector(".layer-clip");
    const handle = frame.querySelector(".compare-handle");
    const set = (pct) => {
      const p = Math.min(100, Math.max(0, pct));
      clipped.style.setProperty("--pos", p + "%");
      handle.style.left = p + "%";
      frame.setAttribute("aria-valuenow", Math.round(p));
    };
    set(50);

    let dragging = false;
    const fromEvent = (e) => {
      const r = frame.getBoundingClientRect();
      set(((e.clientX - r.left) / r.width) * 100);
    };
    frame.addEventListener("pointerdown", (e) => {
      dragging = true;
      frame.setPointerCapture(e.pointerId);
      fromEvent(e);
    });
    frame.addEventListener("pointermove", (e) => dragging && fromEvent(e));
    frame.addEventListener("pointerup", () => (dragging = false));
    frame.addEventListener("pointercancel", () => (dragging = false));

    // Keyboard equivalent, so the comparison is not drag-only.
    frame.tabIndex = 0;
    frame.setAttribute("role", "slider");
    frame.setAttribute("aria-label", `${labelBefore} compared with ${labelAfter}`);
    frame.setAttribute("aria-valuemin", "0");
    frame.setAttribute("aria-valuemax", "100");
    frame.addEventListener("keydown", (e) => {
      const step = e.shiftKey ? 10 : 4;
      const current = parseFloat(frame.getAttribute("aria-valuenow") || "50");
      if (e.key === "ArrowLeft") set(current - step);
      else if (e.key === "ArrowRight") set(current + step);
      else if (e.key === "Home") set(0);
      else if (e.key === "End") set(100);
      else return;
      e.preventDefault();
    });
  });


  /* ── frame player ───────────────────────────────────────────────── */
  document.querySelectorAll(".player").forEach((host) => {
    const prefix = host.dataset.prefix;
    const count = parseInt(host.dataset.count, 10);
    const labels = (host.dataset.labels || "").split(",");
    const coverage = (host.dataset.coverage || "").split(",");
    const alt = host.dataset.alt || "";
    if (!prefix || !count) return;

    const pad = (n) => String(n).padStart(2, "0");
    const frames = Array.from({ length: count }, (_, i) => {
      const img = document.createElement("img");
      img.src = `${prefix}${pad(i + 1)}.jpg`;
      img.alt = `${alt} — ${labels[i] || i + 1}`;
      img.loading = i < 2 ? "eager" : "lazy";
      img.className = "player-frame";
      return img;
    });

    const stage = document.createElement("div");
    stage.className = "player-stage";
    frames.forEach((f) => stage.append(f));

    const controls = document.createElement("div");
    controls.className = "player-controls";
    controls.innerHTML = `
      <button class="player-btn" type="button" aria-label="Play">&#9654;</button>
      <input class="player-range" type="range" min="0" max="${count - 1}" value="0"
             aria-label="Month" />
      <span class="player-readout"><b class="player-label"></b><span class="player-cov"></span></span>`;

    host.append(stage, controls);

    const button = controls.querySelector(".player-btn");
    const range = controls.querySelector(".player-range");
    const label = controls.querySelector(".player-label");
    const cov = controls.querySelector(".player-cov");

    let index = 0;
    let timer = null;

    const show = (i) => {
      index = (i + count) % count;
      frames.forEach((f, n) => f.classList.toggle("on", n === index));
      range.value = index;
      label.textContent = labels[index] || "";
      cov.textContent = coverage[index] ? `${coverage[index]}% ground` : "";
    };

    const stop = () => {
      clearInterval(timer);
      timer = null;
      button.innerHTML = "&#9654;";
      button.setAttribute("aria-label", "Play");
    };
    const play = () => {
      if (timer) return;
      timer = setInterval(() => show(index + 1), 900);
      button.innerHTML = "&#10073;&#10073;";
      button.setAttribute("aria-label", "Pause");
    };

    button.addEventListener("click", () => (timer ? stop() : play()));
    range.addEventListener("input", () => {
      stop();
      show(parseInt(range.value, 10));
    });

    show(0);

    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!still && "IntersectionObserver" in window) {
      let started = false;
      const io = new IntersectionObserver(
        (entries) => {
          entries.forEach((e) => {
            if (e.isIntersecting && !started) {
              started = true;
              play();
            } else if (!e.isIntersecting) {
              stop();
            }
          });
        },
        { threshold: 0.35 }
      );
      io.observe(host);
    }
  });

  /* ── coverage chart ─────────────────────────────────────────────── */
  // June 2025, tile 31UDQ. usable = share of the scene that date could contribute;
  // filled = cumulative share of the mosaic reconstructed. Source: inference_pipeline.log
  // Dates are in processing order — ranked cleanest first, not chronological.
  const DATA = [
    { d: "06-16", usable: 44.9, filled: 44.9 },
    { d: "06-13", usable: 87.8, filled: 92.9 },
    { d: "06-21", usable: 52.8, filled: 94.5 },
    { d: "06-18", usable: 84.3, filled: 94.9 },
    { d: "06-11", usable: 53.7, filled: 95.1 },
    { d: "06-06", usable: 31.2, filled: 95.1 },
    { d: "06-26", usable: 10.4, filled: 95.2 },
    { d: "06-28", usable: 3.1, filled: 95.2 },
    { d: "06-08", usable: 18.6, filled: 95.2 },
    { d: "06-03", usable: 12.2, filled: 95.3 },
    { d: "06-15", usable: 13.1, filled: 95.3 },
    { d: "06-01", usable: 6.4, filled: 95.3 },
    { d: "06-23", usable: 8.4, filled: 95.3 },
    { d: "06-25", usable: 18.3, filled: 95.3 },
    { d: "06-05", usable: 4.2, filled: 95.3 }
  ];

  const host = document.getElementById("coverage-chart");
  if (host) {
    const W = 760, H = 300;
    const M = { t: 14, r: 16, b: 34, l: 34 };
    const iw = W - M.l - M.r, ih = H - M.t - M.b;
    const step = iw / DATA.length;
    const barW = Math.min(20, step * 0.44);
    const x = (i) => M.l + step * (i + 0.5);
    const y = (v) => M.t + ih - (v / 100) * ih;

    const linePath = DATA.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.filled).toFixed(1)}`).join("");

    const gridY = [0, 25, 50, 75, 100];
    const svg = `
    <div class="chart-legend">
      <span><i class="swatch" style="background:var(--series-2)"></i>Usable that date</span>
      <span><i class="swatch line-swatch" style="background:var(--series-1)"></i>Mosaic coverage (cumulative)</span>
    </div>
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
      ${gridY.map((v) => `<line class="grid-line" x1="${M.l}" x2="${W - M.r}" y1="${y(v)}" y2="${y(v)}"></line>
        <text class="axis-text" x="${M.l - 8}" y="${y(v) + 3.5}" text-anchor="end">${v}%</text>`).join("")}
      ${DATA.map((p, i) => `<rect class="bar" x="${(x(i) - barW / 2).toFixed(1)}" y="${y(p.usable).toFixed(1)}"
        width="${barW.toFixed(1)}" height="${(ih - (y(p.usable) - M.t)).toFixed(1)}" rx="3"></rect>`).join("")}
      <path class="line" d="${linePath}"></path>
      ${DATA.map((p, i) => `<circle class="dot" cx="${x(i).toFixed(1)}" cy="${y(p.filled).toFixed(1)}" r="3.5"></circle>`).join("")}
      <text class="label-direct" x="${x(6).toFixed(1)}" y="${(y(94.2) - 12).toFixed(1)}" text-anchor="middle">94.2%</text>
      <text class="label-direct" x="${x(3).toFixed(1)}" y="${(y(48.2) - 12).toFixed(1)}" text-anchor="middle">48.2%</text>
      ${DATA.map((p, i) => `<text class="axis-text" x="${x(i).toFixed(1)}" y="${H - M.b + 18}" text-anchor="middle">${p.d.slice(3)}</text>`).join("")}
      <text class="axis-text" x="${M.l}" y="${H - 4}">June 2025</text>
      <line class="cursor" id="chart-cursor" y1="${M.t}" y2="${M.t + ih}"></line>
    </svg>
    <div class="chart-tip" id="chart-tip"></div>
    <details class="chart-table">
      <summary>Show the numbers</summary>
      <table>
        <thead><tr><th>Date</th><th>Usable</th><th>Coverage</th></tr></thead>
        <tbody>${DATA.map((p) => `<tr><td>${p.d}</td><td>${p.usable.toFixed(1)}%</td><td>${p.filled.toFixed(1)}%</td></tr>`).join("")}</tbody>
      </table>
    </details>`;

    host.innerHTML = svg;

    const svgEl = host.querySelector("svg");
    const cursor = host.querySelector("#chart-cursor");
    const tip = host.querySelector("#chart-tip");
    const bars = [...host.querySelectorAll(".bar")];

    const hide = () => {
      cursor.classList.remove("on");
      tip.style.opacity = 0;
      bars.forEach((b) => b.classList.remove("dim"));
    };

    svgEl.addEventListener("pointermove", (e) => {
      const r = svgEl.getBoundingClientRect();
      const sx = ((e.clientX - r.left) / r.width) * W;
      const i = Math.max(0, Math.min(DATA.length - 1, Math.floor((sx - M.l) / step)));
      const p = DATA[i];
      cursor.setAttribute("x1", x(i));
      cursor.setAttribute("x2", x(i));
      cursor.classList.add("on");
      bars.forEach((b, j) => b.classList.toggle("dim", j !== i));
      tip.innerHTML = `<b>2025-${p.d}</b><br><span class="k">usable</span> ${p.usable.toFixed(1)}%<br><span class="k">coverage</span> ${p.filled.toFixed(1)}%`;
      tip.style.left = (r.left - host.getBoundingClientRect().left) + (x(i) / W) * r.width + "px";
      tip.style.top = (r.top - host.getBoundingClientRect().top) + (y(Math.max(p.filled, p.usable)) / H) * r.height + "px";
      tip.style.opacity = 1;
    });
    svgEl.addEventListener("pointerleave", hide);
  }
})();

"use strict";

const $ = (id) => document.getElementById(id);

let meta = null;        // {n,w,h,cw,x_max,boxes,feasible}
let images = [];        // 每帧 Image
let trajectory = null;  // 求解成功后的逐帧 x
let locks = {};         // 帧号 -> 锁定 x
let cur = 0;
let shownX = 0;         // 当前帧画面上显示的裁切窗横坐标
let failFrame = null;   // 求解失败时最早接不上的帧
let dragging = false;
let grabDX = 0;
let loadToken = 0;      // 会话版本，用于帧图片缓存刷新

// ------------------------------------------------------------ 状态提示

function setStatus(msg, kind) {
  const el = $("status");
  el.textContent = msg;
  el.className = "status " + (kind || "");
}

// ------------------------------------------------------------ 数据加载

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  return handle(r);
}

async function handle(r) {
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || ("HTTP " + r.status));
  return data;
}

function naturalSort(files) {
  return [...files].sort((a, b) =>
    a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" }));
}

async function doUpload() {
  const fs = naturalSort($("fileFrames").files);
  const bf = $("fileBoxes").files[0];
  if (!fs.length || !bf) { setStatus("请选择 PNG 帧和必留框 JSON", "err"); return; }
  const fd = new FormData();
  for (const f of fs) fd.append("frames", f, f.name);
  fd.append("boxes", bf, bf.name);
  setStatus("上传中…");
  try {
    const r = await fetch("/api/upload", { method: "POST", body: fd });
    await afterLoad(await handle(r));
  } catch (e) { setStatus("上传失败：" + e.message, "err"); }
}

async function loadSample(name) {
  setStatus("载入样片…");
  try {
    const r = await fetch("/api/load_sample?name=" + name, { method: "POST" });
    await afterLoad(await handle(r));
  } catch (e) { setStatus("载入样片失败：" + e.message, "err"); }
}

async function afterLoad(m) {
  meta = m;
  locks = {};
  trajectory = null;
  failFrame = null;
  cur = 0;
  shownX = 0;
  loadToken++;
  images = await Promise.all([...Array(m.n).keys()].map(loadFrame));
  $("slider").max = m.n - 1;
  $("slider").value = 0;
  $("cvMain").width = m.w; $("cvMain").height = m.h;
  $("cvPreview").width = m.cw; $("cvPreview").height = m.h;
  $("viewer").hidden = false;
  $("btnSolve").disabled = false;
  setStatus(`已载入 ${m.n} 帧（${m.w}×${m.h}，裁切窗 ${m.cw}×${m.h}，x ∈ [0, ${m.x_max}]），求解中…`);
  await solve();
}

function loadFrame(i) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("帧 " + i + " 加载失败"));
    img.src = "/api/frame/" + i + ".png?t=" + loadToken;
  });
}

// ------------------------------------------------------------ 求解

async function solve() {
  if (!meta) return;
  try {
    const r = await postJSON("/api/solve", { locks });
    if (r.ok) {
      trajectory = r.trajectory;
      failFrame = null;
      shownX = trajectory[cur];
      setStatus(`求解成功，总能量 ${Math.round(r.energy).toLocaleString()}。可拖动裁切窗并锁定。`, "ok");
    } else {
      trajectory = null;
      failFrame = r.fail_frame;
      cur = failFrame;
      $("slider").value = cur;
      shownX = defaultX(cur);
      setStatus(r.message + "——已撤下竖屏预览，可调整锁定后重新求解。", "err");
    }
  } catch (e) {
    setStatus("求解失败：" + e.message, "err");
  }
  render();
}

// ------------------------------------------------------------ 绘制

function render() {
  if (!meta) return;
  const cv = $("cvMain"), ctx = cv.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(images[cur], 0, 0);

  // 必留框（红）
  ctx.lineWidth = 1;
  ctx.strokeStyle = "#e33";
  for (const b of meta.boxes[cur] || []) {
    ctx.strokeRect(b[0] + 0.5, b[1] + 0.5, b[2] - b[0] - 1, b[3] - b[1] - 1);
  }

  const failed = trajectory === null;
  $("previewPane").style.visibility = failed ? "hidden" : "visible";

  // 裁切窗（锁定绿 / 未锁定蓝；失败时也可拖动以便重新锁定）
  const locked = locks[cur] !== undefined;
  ctx.strokeStyle = locked ? "#0a0" : "#28f";
  ctx.lineWidth = 2;
  ctx.strokeRect(shownX + 1, 1, meta.cw - 2, meta.h - 2);
  ctx.fillStyle = locked ? "#0a0" : "#28f";
  ctx.font = "12px sans-serif";
  ctx.fillText((locked ? "🔒 " : "") + "x=" + shownX, Math.min(shownX + 4, meta.w - 60), 14);

  if (!failed) {
    // 竖屏预览
    const pv = $("cvPreview").getContext("2d");
    pv.imageSmoothingEnabled = false;
    pv.drawImage(images[cur], shownX, 0, meta.cw, meta.h, 0, 0, meta.cw, meta.h);
  } else {
    ctx.fillStyle = "rgba(200,0,0,0.85)";
    ctx.font = "bold 14px sans-serif";
    ctx.fillText("第 " + cur + " 帧起无法接续", 8, 18);
  }

  $("frameLabel").textContent = `第 ${cur + 1}/${meta.n} 帧`;
  $("btnLock").textContent = locks[cur] !== undefined ? "更新锁定为当前位置" : "锁定本帧位置";
  renderLocks();
}

function renderLocks() {
  const el = $("locks");
  const keys = Object.keys(locks).map(Number).sort((a, b) => a - b);
  el.innerHTML = keys.length
    ? "已锁定：" + keys.map((i) => `<span class="chip">帧 ${i + 1} → x=${locks[i]}</span>`).join("")
    : "未锁定任何帧。";
}

// ------------------------------------------------------------ 交互

function clampX(x) {
  return Math.max(0, Math.min(meta.x_max, Math.round(x)));
}

function defaultX(i) {
  if (locks[i] !== undefined) return locks[i];
  const f = meta.feasible[i];
  return f ? Math.round((f[0] + f[1]) / 2) : 0;
}

function gotoFrame(i) {
  cur = Math.max(0, Math.min(meta.n - 1, i));
  $("slider").value = cur;
  shownX = trajectory ? trajectory[cur] : defaultX(cur);
  render();
}

$("btnUpload").onclick = doUpload;
$("btnSampleA").onclick = () => loadSample("a");
$("btnSampleB").onclick = () => loadSample("b");
$("btnSolve").onclick = solve;
$("btnPrev").onclick = () => gotoFrame(cur - 1);
$("btnNext").onclick = () => gotoFrame(cur + 1);
$("slider").oninput = (e) => gotoFrame(+e.target.value);

$("btnLock").onclick = () => { locks[cur] = shownX; solve(); };
$("btnUnlock").onclick = () => { delete locks[cur]; solve(); };

const cvMain = $("cvMain");
cvMain.addEventListener("mousedown", (e) => {
  if (!meta) return;
  dragging = true;
  grabDX = e.offsetX - shownX;
});
window.addEventListener("mousemove", (e) => {
  if (!dragging) return;
  const rect = cvMain.getBoundingClientRect();
  shownX = clampX(e.clientX - rect.left - grabDX);
  render();
});
window.addEventListener("mouseup", async () => {
  if (!dragging) return;
  dragging = false;
  if (locks[cur] !== undefined && locks[cur] !== shownX) {
    locks[cur] = shownX;   // 拖动已锁定帧 = 改锁定位置，前后未锁定帧跟着重新求解
    await solve();
  }
});

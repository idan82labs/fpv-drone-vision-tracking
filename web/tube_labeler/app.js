const $ = (id) => document.getElementById(id);

const els = {
  progressText: $("progressText"),
  saveState: $("saveState"),
  prevBtn: $("prevBtn"),
  nextBtn: $("nextBtn"),
  frameSelect: $("frameSelect"),
  unlabeledOnly: $("unlabeledOnly"),
  checkpointTitle: $("checkpointTitle"),
  checkpointMeta: $("checkpointMeta"),
  clearCheckpointBtn: $("clearCheckpointBtn"),
  video: $("video"),
  videoMeta: $("videoMeta"),
  overviewStage: $("overviewStage"),
  overviewImage: $("overviewImage"),
  targetOverlay: $("targetOverlay"),
  targetDraft: $("targetDraft"),
  targetStatus: $("targetStatus"),
  markTargetBtn: $("markTargetBtn"),
  targetVisibleBtn: $("targetVisibleBtn"),
  targetNotVisibleBtn: $("targetNotVisibleBtn"),
  clearTargetBtn: $("clearTargetBtn"),
  cropImage: $("cropImage"),
  cropMeta: $("cropMeta"),
  candidateProgress: $("candidateProgress"),
  activeCandidate: $("activeCandidate"),
  candidateStrip: $("candidateStrip"),
  candidateReview: $("candidateReview"),
  modeHelp: $("modeHelp"),
  overviewLegend: $("overviewLegend"),
};

const labelGroups = [
  { title: "Object labels", ids: ["target", "near_target_wrong_center", "uncertain"] },
  {
    title: "Not target: choose the main reason",
    ids: [
      "static_hotspot",
      "line_attached",
      "parallax_edge",
      "boundary_artifact",
      "appearance_blob",
      "terrain_texture",
      "noise",
    ],
  },
];

let state = null;
let labels = [];
let labelById = new Map();
let currentIndex = 0;
let selectedRowId = null;
let noteTimer = null;
let targetMarking = false;
let targetDrag = null;

function htmlEscape(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function fmtScore(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(2) : "";
}

function fmtTime(seconds) {
  const s = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(s / 60);
  const rest = s - minutes * 60;
  return `${minutes}:${rest.toFixed(2).padStart(5, "0")}`;
}

function checkpoint() {
  if (!state?.checkpoints?.length) return null;
  return state.checkpoints[currentIndex] || state.checkpoints[0];
}

function parseBbox(value) {
  if (!value) return null;
  try {
    const parsed = JSON.parse(value);
    if (!Array.isArray(parsed) || parsed.length !== 4) return null;
    const nums = parsed.map((item) => Number(item));
    if (nums.some((num) => !Number.isFinite(num))) return null;
    const [x, y, w, h] = nums;
    return [x, y, Math.max(1, w), Math.max(1, h)];
  } catch {
    return null;
  }
}

function bboxText(bbox) {
  if (!bbox) return "";
  return JSON.stringify(bbox.map((item) => Math.round(Number(item) || 0)));
}

function isActiveMineCheckpoint(cp) {
  return String(cp?.checkpoint_label || "").includes("active_mine");
}

function selectedRow() {
  const cp = checkpoint();
  if (!cp) return null;
  return cp.rows.find((row) => row.id === selectedRowId) || cp.rows[0] || null;
}

function setSaveState(text, cls = "") {
  els.saveState.className = `saveState ${cls}`;
  els.saveState.textContent = text;
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await res.json();
  if (!res.ok || data.ok === false) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

async function loadState() {
  state = await fetchJson("/api/state");
  labels = state.labels;
  labelById = new Map(labels.map((label) => [label.id, label]));
  currentIndex = Number(localStorage.getItem("tubeLabelerIndex") || "0");
  currentIndex = Math.max(0, Math.min(currentIndex, state.checkpoints.length - 1));
  selectedRowId = checkpoint()?.rows?.[0]?.id ?? null;
  renderAll();
}

function renderAll() {
  renderStats();
  renderFrameSelect();
  renderCheckpoint();
}

function renderStats() {
  const p = state.progress;
  els.progressText.textContent = `${p.labeled_rows}/${p.total_rows} candidates labeled`;
}

function renderFrameSelect() {
  const frag = document.createDocumentFragment();
  state.checkpoints.forEach((item, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `Frame ${index + 1}/${state.checkpoints.length} | ${item.clip_short} | ${fmtTime(item.time_seconds)} | ${item.labeled_count}/${item.row_count}`;
    frag.appendChild(option);
  });
  els.frameSelect.replaceChildren(frag);
  els.frameSelect.value = String(currentIndex);
}

function renderCheckpoint() {
  const cp = checkpoint();
  if (!cp) return;
  const activeMine = isActiveMineCheckpoint(cp);
  els.modeHelp.textContent = activeMine
    ? "Label each red candidate. If the real drone is visible but missed, use Actual drone > Mark box on the overview."
    : "Label the red candidate. If the real drone position needs correction, use Actual drone > Mark box.";
  els.overviewLegend.textContent = activeMine
    ? "cyan = model-selected candidate, red = selected rank"
    : "cyan = reference drone, red = selected rank";
  const labeled = cp.rows.filter((row) => row.human_label).length;
  els.checkpointTitle.textContent = `Frame ${currentIndex + 1} of ${state.checkpoints.length}`;
  els.checkpointMeta.textContent = `${cp.clip_short} | video time ${fmtTime(cp.time_seconds)} | ${labeled}/${cp.rows.length} candidates labeled`;
  els.candidateProgress.textContent = `${labeled}/${cp.rows.length}`;
  els.videoMeta.textContent = cp.video_exists ? `${Math.round(cp.fps * 100) / 100} fps` : "video missing";
  setVideo(cp);
  renderFrameTargetUI();
  renderCandidateStrip(cp);
  renderCandidateReview();
  updateNavButtons();
}

function setVideo(cp) {
  if (!cp.video_exists) {
    els.video.removeAttribute("src");
    return;
  }
  const seek = () => seekFrame(0);
  if (els.video.dataset.clip !== cp.clip) {
    els.video.dataset.clip = cp.clip;
    els.video.src = cp.video_url;
    els.video.load();
    els.video.onloadedmetadata = seek;
  } else {
    seek();
  }
}

function seekFrame(offsetFrames) {
  const cp = checkpoint();
  if (!cp?.video_exists) return;
  try {
    els.video.pause();
    els.video.currentTime = Math.max(0, cp.frame + Number(offsetFrames || 0)) / cp.fps;
  } catch {
    // loadedmetadata will retry after the media element is ready.
  }
}

function renderCandidateStrip(cp) {
  const frag = document.createDocumentFragment();
  cp.rows.forEach((row) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = ["candidateTab", row.id === selectedRowId ? "active" : "", row.human_label ? "done" : ""].join(" ");
    const label = row.human_label ? labelById.get(row.human_label)?.name || row.human_label : "Unlabeled";
    btn.innerHTML = `<strong>Rank ${row.rank_int}</strong><span>${htmlEscape(label)}</span>`;
    btn.addEventListener("click", () => {
      selectedRowId = row.id;
      renderCandidateStrip(cp);
      renderCandidateReview();
    });
    frag.appendChild(btn);
  });
  els.candidateStrip.replaceChildren(frag);
}

function renderCandidateReview() {
  const row = selectedRow();
  if (!row) {
    els.candidateReview.textContent = "";
    return;
  }
  const label = row.human_label || "";
  const labelName = label ? labelById.get(label)?.name || label : "Unlabeled";
  const activeMine = isActiveMineCheckpoint(checkpoint());
  const summaryText = activeMine
    ? "The red box is the candidate you are labeling. The cyan box is the model-selected high-score candidate that caused this frame to be mined."
    : "The red box is the candidate you are labeling. The cyan box in the overview is the reference drone position.";
  const targetText = activeMine
    ? "Choose Drone target only when the red candidate is actually on a visible drone. Otherwise choose the main clutter reason."
    : "Choose Drone target only when the red candidate is on the cyan/reference drone.";
  els.activeCandidate.textContent = `Rank ${row.rank_int} | ${labelName}`;
  els.overviewImage.src = `${row.selected_overview_url}&t=${Date.now()}`;
  els.overviewImage.onload = renderFrameTargetUI;
  els.cropImage.src = `${row.candidate_crop_url}&t=${Date.now()}`;
  els.cropMeta.textContent = `rank ${row.rank_int} crop`;
  els.candidateReview.innerHTML = `
    <div class="reviewGrid">
      <aside class="reviewSummary">
        <h2>Candidate rank ${row.rank_int}</h2>
        <p>${htmlEscape(summaryText)}</p>
        <p>${htmlEscape(targetText).replace("Drone target", "<strong>Drone target</strong>")}</p>
        <p><strong>Current label:</strong> ${htmlEscape(labelName)}</p>
        <div class="reviewFacts">
          <div class="fact">Box ${htmlEscape(row.bbox || "")}</div>
          <div class="fact">Source ${htmlEscape(row.candidate_source || "")}</div>
          <div class="fact">${htmlEscape(row.notes || "No detector note")}</div>
        </div>
        <details class="detectorDetails">
          <summary>Detector details</summary>
          <div class="detailGrid">
            <div class="fact">verified ${fmtScore(row.verified_score_float)}</div>
            <div class="fact">raw ${fmtScore(row.raw_score_float)}</div>
            <div class="fact">verifier ${fmtScore(row.tube_verifier_score_float)}</div>
          </div>
        </details>
      </aside>
      <div class="labelGroups">
        ${labelGroups.map((group) => renderLabelGroup(group, row)).join("")}
        <button class="labelBtn clearLabel" data-label="" type="button">
          <strong>Clear label</strong><span>Remove this candidate label and keep reviewing.</span>
        </button>
        <div class="notesBlock">
          <label for="candidateNotes">Optional note</label>
          <textarea id="candidateNotes" rows="3" placeholder="Example: drone visible lower right; candidate is pole top">${htmlEscape(row.human_notes || "")}</textarea>
        </div>
      </div>
    </div>
  `;
  els.candidateReview.querySelectorAll("[data-label]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      await saveRow(row.id, btn.dataset.label, undefined);
      selectNextUnlabeledCandidate();
    });
  });
  els.candidateReview.querySelector("#candidateNotes").addEventListener("input", (event) => {
    queueNotesSave(row.id, event.target.value);
  });
}

function imageGeometry() {
  const img = els.overviewImage;
  if (!img?.naturalWidth || !img?.naturalHeight) return null;
  const imageRect = img.getBoundingClientRect();
  const stageRect = els.overviewStage.getBoundingClientRect();
  if (!imageRect.width || !imageRect.height) return null;
  return {
    left: imageRect.left - stageRect.left,
    top: imageRect.top - stageRect.top,
    width: imageRect.width,
    height: imageRect.height,
    naturalWidth: img.naturalWidth,
    naturalHeight: img.naturalHeight,
  };
}

function renderOverlayBox(el, bbox) {
  const geom = imageGeometry();
  if (!geom || !bbox) {
    el.hidden = true;
    return;
  }
  const [x, y, w, h] = bbox;
  el.style.left = `${geom.left + (x / geom.naturalWidth) * geom.width}px`;
  el.style.top = `${geom.top + (y / geom.naturalHeight) * geom.height}px`;
  el.style.width = `${Math.max(4, (w / geom.naturalWidth) * geom.width)}px`;
  el.style.height = `${Math.max(4, (h / geom.naturalHeight) * geom.height)}px`;
  el.hidden = false;
}

function renderFrameTargetUI() {
  const cp = checkpoint();
  if (!cp) return;
  const bbox = parseBbox(cp.frame_target_bbox);
  renderOverlayBox(els.targetOverlay, bbox);
  if (!targetDrag) els.targetDraft.hidden = true;
  const visible = cp.frame_target_visible || "";
  if (bbox) {
    els.targetStatus.textContent = `box ${bboxText(bbox)} in overview coords`;
  } else if (visible === "yes") {
    els.targetStatus.textContent = "visible, no box";
  } else if (visible === "no") {
    els.targetStatus.textContent = "not visible";
  } else if (visible === "uncertain") {
    els.targetStatus.textContent = "uncertain";
  } else {
    els.targetStatus.textContent = "not marked";
  }
  els.markTargetBtn.classList.toggle("active", targetMarking);
}

function pointInOverview(event) {
  const img = els.overviewImage;
  if (!img?.naturalWidth || !img?.naturalHeight) return null;
  const rect = img.getBoundingClientRect();
  const xDisplay = event.clientX - rect.left;
  const yDisplay = event.clientY - rect.top;
  if (xDisplay < 0 || yDisplay < 0 || xDisplay > rect.width || yDisplay > rect.height) return null;
  const x = (xDisplay / rect.width) * img.naturalWidth;
  const y = (yDisplay / rect.height) * img.naturalHeight;
  return {
    x: Math.max(0, Math.min(img.naturalWidth - 1, x)),
    y: Math.max(0, Math.min(img.naturalHeight - 1, y)),
  };
}

function bboxFromPoints(start, end) {
  const img = els.overviewImage;
  const dx = Math.abs(end.x - start.x);
  const dy = Math.abs(end.y - start.y);
  if (dx < 4 && dy < 4) {
    const size = 8;
    return [
      Math.max(0, Math.round(start.x - size / 2)),
      Math.max(0, Math.round(start.y - size / 2)),
      size,
      size,
    ];
  }
  const x0 = Math.max(0, Math.min(start.x, end.x));
  const y0 = Math.max(0, Math.min(start.y, end.y));
  const x1 = Math.min(img.naturalWidth - 1, Math.max(start.x, end.x));
  const y1 = Math.min(img.naturalHeight - 1, Math.max(start.y, end.y));
  return [
    Math.round(x0),
    Math.round(y0),
    Math.max(3, Math.round(x1 - x0)),
    Math.max(3, Math.round(y1 - y0)),
  ];
}

async function saveFrameTarget({ bbox = null, visible = "", notes = undefined } = {}) {
  const cp = checkpoint();
  if (!cp) return;
  try {
    setSaveState("Saving", "saving");
    const selected = selectedRowId;
    const index = currentIndex;
    const payload = {
      clip: cp.clip,
      frame: cp.frame,
      frame_target_bbox: bboxText(bbox),
      frame_target_visible: visible,
      frame_target_notes: notes ?? cp.frame_target_notes ?? "",
    };
    const data = await fetchJson("/api/frame_target", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state = data.state;
    labels = state.labels;
    labelById = new Map(labels.map((item) => [item.id, item]));
    currentIndex = Math.max(0, Math.min(index, state.checkpoints.length - 1));
    selectedRowId = checkpoint()?.rows?.some((row) => row.id === selected)
      ? selected
      : checkpoint()?.rows?.[0]?.id ?? null;
    setSaveState("Saved", "saved");
    renderAll();
  } catch (error) {
    setSaveState(error.message, "error");
  }
}

function beginTargetDrag(event) {
  if (!targetMarking || event.button !== 0) return;
  const point = pointInOverview(event);
  if (!point) return;
  event.preventDefault();
  els.overviewStage.setPointerCapture?.(event.pointerId);
  targetDrag = { pointerId: event.pointerId, start: point, end: point };
  renderOverlayBox(els.targetDraft, bboxFromPoints(point, point));
}

function updateTargetDrag(event) {
  if (!targetDrag || targetDrag.pointerId !== event.pointerId) return;
  const point = pointInOverview(event);
  if (!point) return;
  targetDrag.end = point;
  renderOverlayBox(els.targetDraft, bboxFromPoints(targetDrag.start, targetDrag.end));
}

async function endTargetDrag(event) {
  if (!targetDrag || targetDrag.pointerId !== event.pointerId) return;
  const drag = targetDrag;
  targetDrag = null;
  els.targetDraft.hidden = true;
  els.overviewStage.releasePointerCapture?.(event.pointerId);
  const end = pointInOverview(event) || drag.end;
  targetMarking = false;
  els.overviewStage.classList.remove("marking");
  renderFrameTargetUI();
  await saveFrameTarget({ bbox: bboxFromPoints(drag.start, end), visible: "yes" });
}

function renderLabelGroup(group, row) {
  const buttons = group.ids
    .map((id) => {
      const item = labelById.get(id);
      if (!item) return "";
      const active = row.human_label === id ? "active" : "";
      const target = id === "target" ? "target" : "";
      return `
        <button class="labelBtn ${active} ${target}" data-label="${id}" type="button">
          <strong>${htmlEscape(item.name)}</strong><span>${htmlEscape(item.hint)}</span>
        </button>
      `;
    })
    .join("");
  return `<div class="labelGroup"><div class="labelGroupTitle">${htmlEscape(group.title)}</div><div class="labelButtons">${buttons}</div></div>`;
}

function queueNotesSave(rowId, notes) {
  if (noteTimer) clearTimeout(noteTimer);
  noteTimer = setTimeout(async () => {
    noteTimer = null;
    await saveRow(rowId, undefined, notes, { quiet: true });
  }, 450);
}

async function saveRow(rowId, label, notes, { quiet = false } = {}) {
  try {
    if (!quiet) setSaveState("Saving", "saving");
    const payload = { id: rowId };
    if (label !== undefined) payload.human_label = label;
    if (notes !== undefined) payload.human_notes = notes;
    const data = await fetchJson("/api/label", { method: "POST", body: JSON.stringify(payload) });
    updateLocalRow(data.row);
    state.progress = data.progress;
    renderStats();
    renderFrameSelect();
    renderCandidateStrip(checkpoint());
    renderCandidateReview();
    setSaveState("Saved", "saved");
  } catch (error) {
    setSaveState(error.message, "error");
  }
}

function updateLocalRow(updated) {
  for (const cp of state.checkpoints) {
    const index = cp.rows.findIndex((row) => row.id === updated.id);
    if (index >= 0) {
      cp.rows[index] = { ...cp.rows[index], ...updated };
      cp.labeled_count = cp.rows.filter((row) => row.human_label).length;
      cp.target_count = cp.rows.filter((row) => row.human_label === "target").length;
      cp.complete = cp.labeled_count === cp.rows.length;
      cp.has_target = cp.target_count > 0;
      return;
    }
  }
}

function selectNextUnlabeledCandidate() {
  const cp = checkpoint();
  if (!cp) return;
  const currentPos = cp.rows.findIndex((row) => row.id === selectedRowId);
  const next = cp.rows
    .slice(currentPos + 1)
    .concat(cp.rows.slice(0, currentPos + 1))
    .find((row) => !row.human_label);
  if (next) {
    selectedRowId = next.id;
    renderCandidateStrip(cp);
    renderCandidateReview();
  }
}

async function clearCheckpointLabels() {
  const cp = checkpoint();
  if (!cp) return;
  if (!window.confirm(`Clear labels and notes for frame ${cp.frame}?`)) return;
  try {
    setSaveState("Saving", "saving");
    const data = await fetchJson("/api/clear_checkpoint", {
      method: "POST",
      body: JSON.stringify({ clip: cp.clip, frame: cp.frame }),
    });
    state = data.state;
    labels = state.labels;
    labelById = new Map(labels.map((item) => [item.id, item]));
    selectedRowId = checkpoint()?.rows?.[0]?.id ?? null;
    setSaveState("Saved", "saved");
    renderAll();
  } catch (error) {
    setSaveState(error.message, "error");
  }
}

function updateNavButtons() {
  els.prevBtn.disabled = !state || state.checkpoints.length <= 1;
  els.nextBtn.disabled = !state || state.checkpoints.length <= 1;
}

function persistIndex() {
  localStorage.setItem("tubeLabelerIndex", String(currentIndex));
}

function checkpointVisibleForNav(cp) {
  if (!els.unlabeledOnly.checked) return true;
  return cp.rows.some((row) => !row.human_label);
}

function go(delta) {
  if (!state?.checkpoints?.length) return;
  let index = currentIndex;
  for (let i = 0; i < state.checkpoints.length; i += 1) {
    index = (index + delta + state.checkpoints.length) % state.checkpoints.length;
    if (checkpointVisibleForNav(state.checkpoints[index])) {
      currentIndex = index;
      selectedRowId = state.checkpoints[index].rows[0]?.id ?? null;
      persistIndex();
      renderAll();
      return;
    }
  }
}

function selectCandidateOffset(delta) {
  const cp = checkpoint();
  if (!cp?.rows?.length) return;
  const currentPos = Math.max(0, cp.rows.findIndex((row) => row.id === selectedRowId));
  const nextPos = (currentPos + delta + cp.rows.length) % cp.rows.length;
  selectedRowId = cp.rows[nextPos].id;
  renderCandidateStrip(cp);
  renderCandidateReview();
}

function applyKeyboardLabel(key) {
  const row = selectedRow();
  const label = labels.find((item) => item.key === key);
  if (!row || !label) return false;
  saveRow(row.id, label.id, undefined);
  return true;
}

function wireEvents() {
  els.prevBtn.addEventListener("click", () => go(-1));
  els.nextBtn.addEventListener("click", () => go(1));
  els.frameSelect.addEventListener("change", () => {
    currentIndex = Number(els.frameSelect.value || "0");
    selectedRowId = checkpoint()?.rows?.[0]?.id ?? null;
    persistIndex();
    renderAll();
  });
  els.unlabeledOnly.addEventListener("change", () => renderFrameSelect());
  els.clearCheckpointBtn.addEventListener("click", clearCheckpointLabels);
  els.markTargetBtn.addEventListener("click", () => {
    targetMarking = !targetMarking;
    els.overviewStage.classList.toggle("marking", targetMarking);
    renderFrameTargetUI();
  });
  els.targetVisibleBtn.addEventListener("click", () => saveFrameTarget({ visible: "yes" }));
  els.targetNotVisibleBtn.addEventListener("click", () => saveFrameTarget({ visible: "no" }));
  els.clearTargetBtn.addEventListener("click", () => saveFrameTarget({ visible: "", bbox: null, notes: "" }));
  els.overviewStage.addEventListener("pointerdown", beginTargetDrag);
  els.overviewStage.addEventListener("pointermove", updateTargetDrag);
  els.overviewStage.addEventListener("pointerup", endTargetDrag);
  els.overviewStage.addEventListener("pointercancel", () => {
    targetDrag = null;
    els.targetDraft.hidden = true;
  });
  window.addEventListener("resize", renderFrameTargetUI);
  document.querySelectorAll(".frameControls button").forEach((btn) => {
    btn.addEventListener("click", () => seekFrame(Number(btn.dataset.step)));
  });
  window.addEventListener("keydown", (event) => {
    const active = document.activeElement;
    const inText = active && ["TEXTAREA", "INPUT"].includes(active.tagName);
    if (inText || event.altKey || event.metaKey || event.ctrlKey) return;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      go(-1);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      go(1);
    } else if (event.key === "[") {
      event.preventDefault();
      selectCandidateOffset(-1);
    } else if (event.key === "]") {
      event.preventDefault();
      selectCandidateOffset(1);
    } else if (applyKeyboardLabel(event.key)) {
      event.preventDefault();
    }
  });
}

wireEvents();
loadState().catch((error) => setSaveState(error.message, "error"));

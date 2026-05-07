/* upload.js — drag-and-drop, preview, and form submit handling */

const dropZone   = document.getElementById("dropZone");
const fileInput  = document.getElementById("fileInput");
const dropInner  = document.getElementById("dropInner");
const previewArea= document.getElementById("previewArea");
const previewImg = document.getElementById("previewImg");
const clearBtn   = document.getElementById("clearBtn");
const uploadForm = document.getElementById("uploadForm");
const classifyBtn= document.getElementById("classifyBtn");

// ── Drag events ───────────────────────────────────────────────────────────────
dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
["dragleave", "dragend"].forEach(ev =>
  dropZone.addEventListener(ev, () => dropZone.classList.remove("drag-over"))
);
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});

// Click anywhere on drop zone (but not clear btn) opens picker
dropZone.addEventListener("click", e => {
  if (e.target === clearBtn) return;
  if (!previewArea.style.display || previewArea.style.display === "none") {
    fileInput.click();
  }
});

// File input change
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) setFile(fileInput.files[0]);
});

// Clear
clearBtn.addEventListener("click", e => {
  e.stopPropagation();
  clearPreview();
});

// Submit guard
uploadForm.addEventListener("submit", e => {
  if (!fileInput.files[0]) {
    e.preventDefault();
    alert("Please select an image first.");
    return;
  }
  classifyBtn.disabled = true;
  classifyBtn.querySelector(".btn-classify__text").textContent = "Classifying…";
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function setFile(file) {
  // Validate extension
  const allowed = [".jpg",".jpeg",".png",".bmp",".tiff",".tif"];
  const ext = "." + file.name.split(".").pop().toLowerCase();
  if (!allowed.includes(ext)) {
    alert(`Unsupported file type '${ext}'. Please upload a JPG, PNG, BMP, or TIFF.`);
    return;
  }

  // Transfer to the real input so Flask receives it
  const dt = new DataTransfer();
  dt.items.add(file);
  fileInput.files = dt.files;

  // Show preview
  const reader = new FileReader();
  reader.onload = ev => {
    previewImg.src = ev.target.result;
    dropInner.style.display  = "none";
    previewArea.style.display = "flex";
  };
  reader.readAsDataURL(file);
}

function clearPreview() {
  previewImg.src = "";
  fileInput.value = "";
  previewArea.style.display = "none";
  dropInner.style.display   = "";
}

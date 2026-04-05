import WaveSurfer from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.esm.js'
import RegionsPlugin from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/regions.esm.js'
import Minimap from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/minimap.esm.js'
import Timeline from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/timeline.esm.js'
import { DETECTOR_DOCS } from './docs.js'

const COLORS = {
  clicks:    { bg: 'rgba(248, 81, 73, 0.25)',  solid: '#f85149' },
  noise:     { bg: 'rgba(210, 153, 34, 0.25)', solid: '#d29922' },
  robotic:   { bg: 'rgba(188, 140, 255, 0.25)', solid: '#bc8cff' },
  bandwidth:     { bg: 'rgba(255, 166, 87, 0.25)',  solid: '#ffa657' },
  pitch_contour: { bg: 'rgba(255, 110, 182, 0.25)', solid: '#ff6eb6' },
}

const SUB_KEYS = ['clicks', 'noise', 'robotic', 'bandwidth', 'pitch_contour']

const SCORE_BANDS = [
  { min: 90, label: 'Отлично', desc: 'Студийное качество, артефакты не обнаружены.' },
  { min: 80, label: 'Хорошо', desc: 'Незначительные артефакты, большинство слушателей не заметят.' },
  { min: 65, label: 'Удовлетворительно', desc: 'Заметные проблемы. Пригодно, но есть куда улучшать.' },
  { min: 50, label: 'Посредственно', desc: 'Множественные артефакты. Качество ниже среднего.' },
  { min: 0,  label: 'Плохо', desc: 'Обнаружены серьёзные артефакты. Не подходит для продакшена.' },
]

const DETECTOR_NAMES = {
  clicks:    'Щелчки',
  noise:     'Шум',
  robotic:   'Роботичность',
  bandwidth:     'Полоса частот',
  pitch_contour: 'Контур высоты тона',
}

const DETECTOR_INFO = {
  clicks:    'Импульсные щелчки и хлопки от сбоев вокодера или границ конкатенации.',
  noise:     'Фоновое шипение или повышенный уровень шума из обучающих данных.',
  robotic:   'Монотонная просодия и плоская интонация — узкий диапазон высоты тона, отсутствие естественной вариативности.',
  bandwidth:     'Узкая спектральная полоса — потеря высоких частот из-за кодеков или сжатия.',
  pitch_contour: 'Неестественный контур F0 — плоский, ступенчатый или с периодическим дрожанием высоты тона.',
}

// DOM refs
const fileInput = document.getElementById('fileInput')
const dropZone = document.getElementById('dropZone')
const dropText = document.getElementById('dropText')
const aliasInput = document.getElementById('aliasInput')
const addBtn = document.getElementById('addBtn')
const reanalyzeBtn = document.getElementById('reanalyzeBtn')
const loading = document.getElementById('loading')
const loadingText = document.getElementById('loadingText')
const libraryBody = document.getElementById('libraryBody')
const emptyMsg = document.getElementById('emptyMsg')
const detail = document.getElementById('detail')
const zoomSlider = document.getElementById('zoomSlider')
const playBtn = document.getElementById('playBtn')
const stopBtn = document.getElementById('stopBtn')
const timeDisplay = document.getElementById('timeDisplay')

let selectedFile = null
let wavesurfer = null
let regionsPlugin = null
let activeTypes = new Set(Object.keys(COLORS))
let library = []
let selectedId = null

// --- Drag & drop + file pick ---
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover') })
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'))
dropZone.addEventListener('drop', (e) => {
  e.preventDefault()
  dropZone.classList.remove('dragover')
  if (e.dataTransfer.files.length) pickFile(e.dataTransfer.files[0])
})
dropZone.addEventListener('click', () => fileInput.click())
fileInput.addEventListener('change', (e) => {
  if (e.target.files.length) pickFile(e.target.files[0])
})

function pickFile(file) {
  if (!file) return
  selectedFile = file
  dropText.textContent = file.name
  dropZone.classList.add('has-file')
  if (!aliasInput.value.trim()) {
    aliasInput.value = file.name.replace(/\.[^.]+$/, '')
  }
  addBtn.disabled = false
  aliasInput.focus()
}

// --- Upload ---
addBtn.addEventListener('click', async () => {
  if (!selectedFile) return
  addBtn.disabled = true
  showLoading('Загрузка и анализ...')

  const form = new FormData()
  form.append('file', selectedFile)
  form.append('alias', aliasInput.value.trim() || selectedFile.name.replace(/\.[^.]+$/, ''))

  try {
    const res = await fetch('/api/audio', { method: 'POST', body: form })
    if (!res.ok) {
      const err = await res.json()
      throw new Error(err.detail || res.statusText)
    }
    const entry = await res.json()
    selectedFile = null
    dropText.textContent = 'Перетащите аудиофайл сюда'
    dropZone.classList.remove('has-file')
    aliasInput.value = ''
    fileInput.value = ''
    await refreshLibrary()
    selectEntry(entry.id)
  } catch (e) {
    alert('Ошибка: ' + e.message)
  } finally {
    hideLoading()
    addBtn.disabled = !selectedFile
  }
})

// --- Reanalyze all ---
reanalyzeBtn.addEventListener('click', async () => {
  reanalyzeBtn.disabled = true
  showLoading('Перезапуск анализа всех файлов...')

  try {
    const res = await fetch('/api/reanalyze', { method: 'POST' })
    if (!res.ok) throw new Error(res.statusText)
    const data = await res.json()
    library = data.entries
    renderLibrary()
    if (selectedId) {
      const entry = library.find(e => e.id === selectedId)
      if (entry) renderDetail(entry)
    }
    if (data.errors.length) {
      alert('Ошибки:\n' + data.errors.join('\n'))
    }
  } catch (e) {
    alert('Ошибка: ' + e.message)
  } finally {
    hideLoading()
    reanalyzeBtn.disabled = library.length === 0
  }
})

// --- Library ---
async function refreshLibrary() {
  try {
    const res = await fetch('/api/audio')
    library = await res.json()
  } catch { library = [] }
  renderLibrary()
}

function renderLibrary() {
  libraryBody.innerHTML = ''
  emptyMsg.classList.toggle('hidden', library.length > 0)
  reanalyzeBtn.disabled = library.length === 0

  for (const entry of library) {
    const tr = document.createElement('tr')
    tr.className = 'lib-row' + (entry.id === selectedId ? ' selected' : '')
    tr.dataset.id = entry.id

    const score = entry.analysis ? Math.round(entry.analysis.score) : '—'
    const scoreClass = entry.analysis
      ? (entry.analysis.score >= 80 ? 'score-green' : entry.analysis.score >= 50 ? 'score-yellow' : 'score-red')
      : ''

    let subCells = ''
    for (const key of SUB_KEYS) {
      const s = entry.analysis?.sub_scores?.[key]
      if (s) {
        const v = Math.round(s.score)
        const cls = v >= 80 ? 'score-green' : v >= 50 ? 'score-yellow' : 'score-red'
        subCells += `<td class="lib-sub ${cls}">${v}</td>`
      } else {
        subCells += `<td class="lib-sub">—</td>`
      }
    }

    tr.innerHTML = `
      <td class="lib-alias">${escHtml(entry.alias)}</td>
      <td class="lib-score ${scoreClass}">${score}</td>
      ${subCells}
      <td><button class="btn-danger" data-delete="${entry.id}" title="Удалить">✕</button></td>
    `

    // Click row to select
    tr.addEventListener('click', (e) => {
      if (e.target.closest('[data-delete]')) return
      selectEntry(entry.id)
    })

    // Double-click alias to edit
    const aliasTd = tr.querySelector('.lib-alias')
    aliasTd.addEventListener('dblclick', (e) => {
      e.stopPropagation()
      startAliasEdit(aliasTd, entry)
    })

    // Delete button
    tr.querySelector('[data-delete]').addEventListener('click', async (e) => {
      e.stopPropagation()
      if (!confirm(`Удалить "${entry.alias}"?`)) return
      await fetch(`/api/audio/${entry.id}`, { method: 'DELETE' })
      if (selectedId === entry.id) {
        selectedId = null
        detail.classList.add('hidden')
        if (wavesurfer) { wavesurfer.destroy(); wavesurfer = null }
      }
      await refreshLibrary()
    })

    libraryBody.appendChild(tr)
  }
}

function startAliasEdit(td, entry) {
  const input = document.createElement('input')
  input.type = 'text'
  input.className = 'alias-edit'
  input.value = entry.alias
  td.textContent = ''
  td.appendChild(input)
  input.focus()
  input.select()

  const save = async () => {
    const newAlias = input.value.trim()
    if (newAlias && newAlias !== entry.alias) {
      await fetch(`/api/audio/${entry.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ alias: newAlias }),
      })
      entry.alias = newAlias
    }
    td.textContent = entry.alias
    if (selectedId === entry.id) {
      document.getElementById('detailTitle').textContent = entry.alias
    }
  }

  input.addEventListener('blur', save)
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') input.blur()
    if (e.key === 'Escape') { input.value = entry.alias; input.blur() }
  })
}

function selectEntry(id) {
  selectedId = id
  document.querySelectorAll('.lib-row').forEach(r => {
    r.classList.toggle('selected', r.dataset.id === id)
  })
  const entry = library.find(e => e.id === id)
  if (entry) renderDetail(entry)
}

// --- Detail panel ---
function renderDetail(entry) {
  detail.classList.remove('hidden')
  document.getElementById('detailTitle').textContent = entry.alias

  if (!entry.analysis) {
    document.getElementById('scoreValue').textContent = '—'
    document.getElementById('scoreExplain').textContent = ''
    return
  }

  const data = entry.analysis
  const score = Math.round(data.score)

  const scoreEl = document.getElementById('scoreValue')
  scoreEl.textContent = score
  scoreEl.className = 'score-value ' + (score >= 80 ? 'score-green' : score >= 50 ? 'score-yellow' : 'score-red')

  const band = SCORE_BANDS.find(b => score >= b.min)
  document.getElementById('scoreExplain').innerHTML = `<strong>${band.label}</strong> &mdash; ${band.desc}`

  // Waveform
  if (wavesurfer) wavesurfer.destroy()

  regionsPlugin = RegionsPlugin.create()

  const minimap = Minimap.create({
    container: '#minimap',
    height: 30,
    waveColor: '#21262d',
    progressColor: '#30363d',
    cursorColor: '#58a6ff',
  })

  const timeline = Timeline.create({
    height: 16,
    style: 'color: #8b949e; font-size: 10px',
    secondaryLabelOpacity: 0.5,
  })

  wavesurfer = WaveSurfer.create({
    container: '#waveform',
    waveColor: '#444c56',
    progressColor: '#58a6ff',
    cursorColor: '#e1e4e8',
    height: 140,
    barWidth: 2,
    barGap: 1,
    barRadius: 2,
    minPxPerSec: 100,
    autoScroll: true,
    autoCenter: true,
    plugins: [regionsPlugin, minimap, timeline],
  })

  wavesurfer.load(`/api/audio/${entry.id}/file`)

  wavesurfer.on('play', () => { playBtn.innerHTML = '&#9646;&#9646;' })
  wavesurfer.on('pause', () => { playBtn.innerHTML = '&#9654;' })
  wavesurfer.on('timeupdate', (t) => {
    timeDisplay.textContent = `${fmtTime(t)} / ${fmtTime(wavesurfer.getDuration())}`
  })
  wavesurfer.on('ready', () => {
    timeDisplay.textContent = `0:00.0 / ${fmtTime(wavesurfer.getDuration())}`
    zoomSlider.value = 100
    drawRegions(data)
  })

  activeTypes = new Set(Object.keys(COLORS))
  renderLegend(data)
  renderSubScores(data)
}

// --- Playback controls ---
playBtn.addEventListener('click', () => { if (wavesurfer) wavesurfer.playPause() })
stopBtn.addEventListener('click', () => { if (wavesurfer) wavesurfer.stop() })

// --- Zoom controls ---
zoomSlider.addEventListener('input', () => {
  if (wavesurfer) wavesurfer.zoom(Number(zoomSlider.value))
})

document.getElementById('zoomInBtn').addEventListener('click', () => {
  zoomSlider.value = Math.min(1000, Number(zoomSlider.value) + 50)
  if (wavesurfer) wavesurfer.zoom(Number(zoomSlider.value))
})

document.getElementById('zoomOutBtn').addEventListener('click', () => {
  zoomSlider.value = Math.max(10, Number(zoomSlider.value) - 50)
  if (wavesurfer) wavesurfer.zoom(Number(zoomSlider.value))
})

document.getElementById('zoomFitBtn').addEventListener('click', () => {
  zoomSlider.value = 10
  if (wavesurfer) wavesurfer.zoom(10)
})

document.getElementById('waveform').addEventListener('wheel', (e) => {
  if (!wavesurfer) return
  e.preventDefault()
  const delta = e.deltaY > 0 ? -30 : 30
  zoomSlider.value = Math.max(10, Math.min(1000, Number(zoomSlider.value) + delta))
  wavesurfer.zoom(Number(zoomSlider.value))
}, { passive: false })

// --- Time formatting ---
function fmtTime(sec) {
  const m = Math.floor(sec / 60)
  const s = Math.floor(sec % 60)
  const ms = Math.floor((sec % 1) * 10)
  return `${m}:${String(s).padStart(2, '0')}.${ms}`
}

// --- Draw artifact regions ---
function drawRegions(data) {
  if (!regionsPlugin) return
  regionsPlugin.clearRegions()

  for (const [type, info] of Object.entries(data.sub_scores)) {
    if (!activeTypes.has(type)) continue
    const color = COLORS[type] || COLORS.clicks
    for (const r of info.regions) {
      const minDuration = 0.005
      const end = Math.max(r.end, r.start + minDuration)
      regionsPlugin.addRegion({
        start: r.start,
        end: end,
        color: r.severity === 'high' ? color.bg.replace('0.25', '0.5') : color.bg,
        content: '',
        resize: false,
        drag: false,
      })
    }
  }
}

// --- Legend ---
function renderLegend(data) {
  const legend = document.getElementById('legend')
  legend.innerHTML = ''

  for (const [type, c] of Object.entries(COLORS)) {
    const count = data.sub_scores[type]?.regions?.length || 0
    const displayName = DETECTOR_NAMES[type] || type
    const item = document.createElement('button')
    item.className = 'legend-toggle' + (activeTypes.has(type) ? ' active' : '')
    item.innerHTML = `<span class="legend-swatch" style="background:${c.solid}"></span>${displayName} <span class="legend-count">(${count})</span>`
    item.addEventListener('click', () => {
      if (activeTypes.has(type)) {
        activeTypes.delete(type)
        item.classList.remove('active')
      } else {
        activeTypes.add(type)
        item.classList.add('active')
      }
      drawRegions(data)
    })
    legend.appendChild(item)
  }
}

// --- Sub-scores table ---
function renderSubScores(data) {
  const tbody = document.querySelector('#subScoresTable tbody')
  tbody.innerHTML = ''
  for (const [type, info] of Object.entries(data.sub_scores)) {
    const color = COLORS[type]?.solid || '#58a6ff'
    const s = Math.round(info.score)
    const desc = DETECTOR_INFO[type] || ''
    const name = DETECTOR_NAMES[type] || type
    const hasDoc = type in DETECTOR_DOCS
    const infoBtn = hasDoc
      ? `<button class="info-btn" data-doc-type="${type}" title="Методология">?</button>`
      : ''
    tbody.innerHTML += `<tr>
      <td><span class="detector-name">${name} ${infoBtn}</span><span class="detector-desc">${desc}</span></td>
      <td>${s}</td>
      <td class="bar-cell"><div class="bar-bg"><div class="bar-fill" style="width:${s}%;background:${color}"></div></div></td>
    </tr>`
  }

  tbody.querySelectorAll('.info-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation()
      openDocsModal(btn.dataset.docType)
    })
  })
}

// --- Docs modal ---
const docsModal = document.getElementById('docsModal')
const docsModalBody = document.getElementById('docsModalBody')
const docsModalClose = document.getElementById('docsModalClose')

function openDocsModal(type) {
  docsModalBody.innerHTML = DETECTOR_DOCS[type] || ''
  docsModal.classList.remove('hidden')
}

docsModalClose.addEventListener('click', () => docsModal.classList.add('hidden'))
docsModal.addEventListener('click', (e) => {
  if (e.target === docsModal) docsModal.classList.add('hidden')
})

// --- Helpers ---
function escHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
}

function showLoading(text) {
  loadingText.textContent = text
  loading.classList.remove('hidden')
}

function hideLoading() {
  loading.classList.add('hidden')
}

// --- Init ---
refreshLibrary()

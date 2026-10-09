const stepContent = {
  read: {
    number: '01', title: 'Value-aware retrieval',
    copy: 'Retrieve skills using semantic relevance, learned utility, and memory strength — so the agent sees experience that is useful for the current clinical context.',
    tags: ['similarity', 'utility', 'memory strength']
  },
  write: {
    number: '02', title: 'Trajectory-to-skill writing',
    copy: 'Turn informative interaction trajectories into structured, reusable procedures across general, task-level, and action-level branches.',
    tags: ['distill', 'draft', 'review']
  },
  assess: {
    number: '03', title: 'Feedback-driven valuation',
    copy: 'Use environment feedback to estimate context-dependent utility, giving the memory a signal for which experiences help and which do not.',
    tags: ['outcome feedback', 'credit', 'risk gate']
  },
  govern: {
    number: '04', title: 'Repository governance',
    copy: 'Promote useful skills, merge redundancy, enforce branch capacity, and remove low-quality or harmful entries as the repository evolves.',
    tags: ['promote', 'merge', 'deprecate']
  }
};

const resultViews = {
  id: {
    eyebrow: 'DEEPSEEK-V3.2 · OFFLINE ID', title: 'Seven-benchmark average',
    legend: ['ReAct', 'SkeMex'],
    bars: [{ label: 'ReAct', value: 48.20, type: 'baseline' }, { label: 'SkeMex', value: 56.08, type: 'skemex' }],
    foot: 'SkeMex improves the average from 48.20% to 56.08% (+7.88 points).',
    callout: 'Useful memory compounds.',
    copy: 'The gain appears while the skill repository is frozen during evaluation, isolating the value of structured experience.',
    anchor: '#S4.SS2'
  },
  ood: {
    eyebrow: 'DEEPSEEK-V3.2 · OFFLINE OOD', title: 'Five unseen benchmark families',
    legend: ['ReAct', 'SkeMex'],
    bars: [{ label: 'ReAct', value: 62.01, type: 'baseline' }, { label: 'SkeMex', value: 75.79, type: 'skemex' }],
    foot: 'The frozen skill repository transfers from 62.01% to 75.79% (+13.78 points).',
    callout: 'Transfer stays stable.',
    copy: 'SkeMex remains above ReAct across the five unseen benchmark families, reducing negative transfer from less structured memories.',
    anchor: '#S5.SS1'
  },
  online: {
    eyebrow: 'DEEPSEEK-V3.2 · ONLINE STREAM', title: 'Continual improvement',
    legend: ['epoch@1', 'epoch@3'],
    bars: [{ label: 'epoch@1', value: 76.39, type: 'baseline' }, { label: 'epoch@3', value: 78.56, type: 'skemex' }],
    foot: 'Online performance rises from 76.39% at epoch@1 to 78.56% at epoch@3 (+2.17 points).',
    callout: 'The loop keeps learning.',
    copy: 'Selective buffering, value-aware retrieval, and utility-based governance reinforce useful procedures over three rounds.',
    anchor: '#S4.SS2'
  },
  transfer: {
    eyebrow: 'THREE BACKBONES · 18 DATASET PAIRS', title: 'Pooled transfer average',
    legend: ['ReAct', 'SkeMex'],
    bars: [{ label: 'ReAct', value: 48.88, type: 'baseline' }, { label: 'SkeMex', value: 59.13, type: 'skemex' }],
    foot: 'Across three additional backbones, the pooled average moves from 48.88% to 59.13% (+10.25 points).',
    callout: 'Skills are portable.',
    copy: 'The advantage persists when the underlying model changes, suggesting that the repository captures procedures rather than one backbone’s response style.',
    anchor: '#S5.SS2'
  }
};

const $ = (selector) => document.querySelector(selector);

function setLifecycleStep(key) {
  const data = stepContent[key];
  document.querySelectorAll('.lifecycle-tab').forEach((tab) => {
    const active = tab.dataset.step === key;
    tab.classList.toggle('is-active', active);
    tab.setAttribute('aria-selected', active ? 'true' : 'false');
  });
  $('.detail-number').textContent = data.number;
  $('#step-title').textContent = data.title;
  $('#step-copy').textContent = data.copy;
  $('#step-tags').innerHTML = data.tags.map((tag) => `<span>${tag}</span>`).join('');
}

function renderChart(key) {
  const data = resultViews[key];
  const chart = $('#bar-chart');
  const max = Math.max(...data.bars.map((bar) => bar.value), 80);
  chart.innerHTML = data.bars.map((bar) => {
    const height = Math.max(7, (bar.value / max) * 100);
    return `<div class="bar-group"><div class="bar ${bar.type}" style="height:${height}%" tabindex="0" role="button" aria-label="${bar.label}: ${bar.value.toFixed(2)} percent" title="${bar.label}: ${bar.value.toFixed(2)}%"><span class="bar-value">${bar.value.toFixed(2)}</span><span class="bar-label">${bar.label}</span></div></div>`;
  }).join('');
  $('#chart-eyebrow').textContent = data.eyebrow;
  $('#chart-title').textContent = data.title;
  $('#chart-foot').textContent = data.foot;
  $('#callout-title').textContent = data.callout;
  $('#callout-copy').textContent = data.copy;
  $('.result-callout a').href = `https://arxiv.org/html/2606.09365v3${data.anchor}`;
  const legend = document.querySelector('.chart-legend');
  legend.innerHTML = data.legend.map((item, index) => `<span><i class="legend-dot ${index ? 'skemex' : 'baseline'}"></i> ${item}</span>`).join('');
}

document.querySelectorAll('.lifecycle-tab').forEach((tab) => tab.addEventListener('click', () => setLifecycleStep(tab.dataset.step)));
document.querySelectorAll('.result-tab').forEach((tab) => tab.addEventListener('click', () => {
  document.querySelectorAll('.result-tab').forEach((item) => {
    const active = item === tab;
    item.classList.toggle('is-active', active);
    item.setAttribute('aria-selected', active ? 'true' : 'false');
  });
  renderChart(tab.dataset.view);
}));

function resolveRepositoryLinks() {
  if (!window.location.hostname.endsWith('.github.io')) return;
  const owner = window.location.hostname.split('.')[0];
  const segments = window.location.pathname.split('/').filter(Boolean);
  if (!owner || !segments[0]) return;
  const repository = `https://github.com/${owner}/${segments[0]}`;
  document.querySelectorAll('[data-repo-link]').forEach((link) => {
    link.href = `${repository}/blob/main/${link.dataset.repoLink}`;
  });
}

const revealItems = document.querySelectorAll('.signal-card,.finding-card,.results-shell,.ablation-chart,.resource-box');
if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver((entries) => entries.forEach((entry) => {
    if (entry.isIntersecting) { entry.target.classList.add('is-visible'); observer.unobserve(entry.target); }
  }), { threshold: .12 });
  revealItems.forEach((item) => { item.classList.add('reveal'); observer.observe(item); });
}

setLifecycleStep('read');
renderChart('id');
resolveRepositoryLinks();

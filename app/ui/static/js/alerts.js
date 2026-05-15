async function loadStats() {
    try {
        const resp = await fetch('/api/alerts/stats');
        const data = await resp.json();
        document.getElementById('stat-total').textContent = data.total_events ?? 0;
        document.getElementById('stat-pending').textContent = data.pending ?? 0;
        document.getElementById('stat-confirmed').textContent = data.confirmed ?? 0;
        document.getElementById('stat-dismissed').textContent = data.dismissed ?? 0;
        document.getElementById('stat-fpr').textContent = `${(((data.false_positive_rate ?? 0) * 100).toFixed(1))}%`;
    } catch (e) {
        console.error('Failed to load alert stats:', e);
    }
}

async function loadEvents() {
    try {
        const status = document.getElementById('filter-status').value;
        const url = status ? `/api/alerts/events?status=${encodeURIComponent(status)}` : '/api/alerts/events';
        const resp = await fetch(url);
        const data = await resp.json();
        const tbody = document.getElementById('events-body');

        if (!data.events || data.events.length === 0) {
            tbody.innerHTML = '<tr><td colspan="10" style="text-align:center;color:#999;">暂无告警事件</td></tr>';
            return;
        }

        tbody.innerHTML = data.events.map((event) => `
            <tr>
                <td>${event.id}</td>
                <td>${escapeHtml(event.alert_type)}</td>
                <td>${thumbnailCell(event)}</td>
                <td>${severityLabel(event.severity)}</td>
                <td>${Number(event.score).toFixed(1)}</td>
                <td>${event.start_frame} - ${event.end_frame}</td>
                <td>${Number(event.duration_seconds ?? 0).toFixed(2)} s</td>
                <td><span class="alert-badge badge-${event.status}">${statusLabel(event.status)}</span></td>
                <td>${formatTime(event.created_at)}</td>
                <td>${actionButtons(event)}</td>
            </tr>
        `).join('');
    } catch (e) {
        console.error('Failed to load alert events:', e);
    }
}

function thumbnailCell(event) {
    if (!event.thumbnail_url) {
        return '<span class="thumb-empty">无抓拍</span>';
    }
    const url = escapeHtml(event.thumbnail_url);
    return `<a href="${url}" target="_blank" rel="noopener"><img class="alert-thumb" src="${url}" alt="吸烟者抓拍"></a>`;
}

function statusLabel(status) {
    const labels = { pending: '待处理', confirmed: '已确认', dismissed: '已忽略' };
    return labels[status] || escapeHtml(status || '-');
}

function severityLabel(severity) {
    const labels = { confirmed: '确认吸烟', suspected: '疑似吸烟', low_confidence: '低置信度' };
    return labels[severity] || escapeHtml(severity || '-');
}

function formatTime(iso) {
    if (!iso) return '-';
    const date = new Date(iso);
    return date.toLocaleString('zh-CN', {
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
    });
}

function actionButtons(event) {
    if (event.status !== 'pending') return '-';
    return `
        <button class="btn-sm btn-confirm" onclick="confirmEvent(${event.id})">确认</button>
        <button class="btn-sm btn-dismiss" onclick="dismissEvent(${event.id})">忽略</button>
    `;
}

async function confirmEvent(id) {
    await fetch(`/api/alerts/events/${id}/confirm`, { method: 'PUT' });
    await refreshAlerts();
}

async function dismissEvent(id) {
    await fetch(`/api/alerts/events/${id}/dismiss`, { method: 'PUT' });
    await refreshAlerts();
}

async function loadRules() {
    try {
        const resp = await fetch('/api/alerts/rules');
        const rules = await resp.json();
        const container = document.getElementById('rules-list');

        if (!rules || rules.length === 0) {
            container.innerHTML = '<p class="muted">暂无告警规则</p>';
            return;
        }

        container.innerHTML = rules.map((rule) => `
            <div class="rule-item">
                <div>
                    <strong>${escapeHtml(rule.name)}</strong>
                    <span class="muted" style="font-size:13px;margin-left:8px;">
                        阈值: ${Number(rule.score_threshold).toFixed(1)} |
                        最小帧数: ${rule.min_duration_frames} |
                        冷却: ${rule.cooldown_seconds}s
                    </span>
                </div>
                <span class="alert-badge ${rule.enabled ? 'badge-confirmed' : 'badge-dismissed'}">
                    ${rule.enabled ? '启用' : '禁用'}
                </span>
            </div>
        `).join('');
    } catch (e) {
        console.error('Failed to load alert rules:', e);
    }
}

async function refreshAlerts() {
    await Promise.all([loadStats(), loadEvents(), loadRules()]);
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

document.addEventListener('DOMContentLoaded', refreshAlerts);

    // -----------------------------------------------------------------------
    // Method metadata — keyed by method.name, built from METHODS constant.
    // -----------------------------------------------------------------------
    var _methodMap = {};
    METHODS.forEach(function(m) { _methodMap[m.name] = m; });

    // Pre-compute unique values across ALL runs (used for conditional pie visibility).
    function _allUnique(key) {
      var seen = {};
      DATA.forEach(function(r) { seen[String(r[key])] = true; });
      return Object.keys(seen);
    }
    var _allUniqueM        = _allUnique('m');
    var _allUniqueTask     = _allUnique('task');
    var _allUniqueEnsemble = _allUnique('ensemble');
    function methodLabel(name) {
      return (_methodMap[name] && _methodMap[name].label) || name || 'unknown';
    }
    var _scoreColors = ['#2ca02c','#1f77b4','#ff7f0e','#9467bd','#8c564b','#e377c2'];
    function methodColor(name) {
      var idx = (SCORES.methods || []).indexOf(name);
      return _scoreColors[idx >= 0 ? idx % _scoreColors.length : 0];
    }
    function fmtTime(s) {
      if (s == null || isNaN(s)) return '?';
      var abs = Math.abs(s);
      if (abs < 60)         return s.toFixed(4) + ' s';
      if (abs < 3600)       return (s / 60).toFixed(3) + ' min';
      if (abs < 86400)      return (s / 3600).toFixed(3) + ' hr';
      if (abs < 31536000)   return (s / 86400).toFixed(3) + ' days';
      return (s / 31536000).toFixed(3) + ' yr';
    }
    function renderMethodBadge(m, val, maxScore) {
      var win = val >= maxScore - 0.001;
      var color = methodColor(m);
      var ns = val === 0 && !win;
      var label = methodLabel(m) + ': ' + (ns ? 'N/A' : val.toFixed(1) + (win ? ' \u2605' : ''));
      return '<span class="msb-badge" style="background:' + color +
             ';color:#fff;opacity:' + (win ? '1' : '0.7') + ';border:1px solid ' + color + '">' +
             label + '</span>';
    }

    // Scoring — pre-computed by Python, embedded as SCORES constant.
    // SCORES = { methods: [...], overall: { scores: {method: avg}, n }, by_mission: {...} }
    // -----------------------------------------------------------------------

    function renderScoreboard() {
      var el = document.getElementById('scoreboard');
      if (!SCORES.overall) { el.style.display = 'none'; return; }

      function scoreCell(scoresObj, methodName) {
        if (!scoresObj || !scoresObj.scores) return '<td class="score-cell" style="color:#bbb">\u2014</td>';
        var val = scoresObj.scores[methodName];
        if (val === undefined) return '<td class="score-cell" style="color:#bbb">\u2014</td>';
        var maxVal = Math.max.apply(null, Object.keys(scoresObj.scores).map(function(m){ return scoresObj.scores[m]; }));
        var isWin = val >= maxVal - 0.001;
        var isNs = val === 0 && !isWin;
        var color = methodColor(methodName);
        return '<td class="score-cell' + (isWin ? ' winner' : '') + '">' +
               '<span style="color:' + color + (isNs ? ';opacity:0.5' : '') + '">' +
               (isNs ? 'N/A' : val.toFixed(1) + (isWin ? ' \u2605' : '')) + '</span>' +
               '</td>';
      }

      // --- Left: overall summary table ---
      var methods = SCORES.methods || [];
      var leftHtml = '<div class="scoreboard-title">Score Summary</div>';
      leftHtml += '<table class="score-table"><thead><tr>';
      leftHtml += '<th></th><th>Overall (' + SCORES.overall.n + ' runs)</th>';
      leftHtml += '</tr></thead><tbody>';
      methods.forEach(function(m) {
        leftHtml += '<tr><td class="team-name" style="color:' + methodColor(m) + '">' + methodLabel(m) + '</td>';
        leftHtml += scoreCell(SCORES.overall, m);
        leftHtml += '</tr>';
      });
      leftHtml += '</tbody></table>';

      // --- Right: filter sliders + live filtered score ---
      var SB_PARAMS = ['n', 'm', 'D'];
      var slidersHtml = '<div class="scoreboard-title">Filtered Score</div><div class="sb-sliders">';
      SB_PARAMS.forEach(function(p) {
        var vals = getUnique(p);
        var maxI = vals.length - 1;
        slidersHtml +=
          '<div class="ar-filter-group">' +
          '<label>' + p + ':</label>' +
          '<div class="dual-range-wrap">' +
            '<div class="dual-range-slider">' +
              '<div class="drs-track"></div>' +
              '<div class="drs-fill" id="sb-drs-' + p + '-fill"></div>' +
              '<input type="range" id="sb-drs-' + p + '-lo" min="0" max="' + maxI + '" value="0">' +
              '<input type="range" id="sb-drs-' + p + '-hi" min="0" max="' + maxI + '" value="' + maxI + '">' +
            '</div>' +
            '<div class="drs-vals">' +
              '<span id="sb-drs-' + p + '-lo-val">' + vals[0] + '</span>' +
              '<span id="sb-drs-' + p + '-hi-val">' + vals[maxI] + '</span>' +
            '</div>' +
          '</div></div>';
      });
      slidersHtml += '</div>';
      var sbTaskVals = getUnique('task');
      slidersHtml += '<div class="filter-row"><span class="fr-label">Tasks:</span><div class="filter-chips">';
      sbTaskVals.forEach(function(t) {
        slidersHtml += '<label class="filter-chip"><input type="checkbox" class="sb-task-cb" value="' + t + '" checked> ' + taskDisplayName(t) + '</label>';
      });
      slidersHtml += '</div></div>';
      var sbMethodVals = SCORES.methods || [];
      slidersHtml += '<div class="filter-row"><span class="fr-label">Methods:</span><div class="filter-chips">';
      sbMethodVals.forEach(function(m) {
        var color = methodColor(m);
        slidersHtml += '<label class="filter-chip" style="border-color:' + color + ';color:' + color + '">' +
          '<input type="checkbox" class="sb-method-cb" value="' + m + '" checked style="accent-color:' + color + '"> ' +
          methodLabel(m) + '</label>';
      });
      slidersHtml += '</div></div>';
      slidersHtml += '<div class="filter-row"><label class="remove-fast-wrap"><input type="checkbox" id="sb-remove-fast"> Runtime &gt; 10s</label></div>';
      slidersHtml += '<div id="sb-filtered-score" class="sb-filtered-score"></div>';

      el.innerHTML =
        '<div class="sb-left">' + leftHtml + '</div>' +
        '<div class="sb-divider"></div>' +
        '<div class="sb-right">' + slidersHtml + '</div>' +
        '<div class="sb-note">\u2605\u2009Score for each run: (winner\u2009time\u00a0/\u00a0my\u2009time)\u00a0\u00d7\u00a0100. The overall score is the average across all runs.</div>';

      // --- Wire up slider logic ---
      var sbVals = {}, sbState = {};
      SB_PARAMS.forEach(function(p) {
        sbVals[p]  = getUnique(p);
        sbState[p] = { lo: 0, hi: sbVals[p].length - 1 };
      });

      function updateSbFill(p) {
        var n = sbVals[p].length;
        var fill = document.getElementById('sb-drs-' + p + '-fill');
        if (!fill || n <= 1) return;
        var pct = 100 / (n - 1);
        fill.style.left  = (sbState[p].lo * pct) + '%';
        fill.style.width = ((sbState[p].hi - sbState[p].lo) * pct) + '%';
      }

      function computeScoreFromRows(rows, removeFast) {
        var groups = {};
        rows.forEach(function(r) {
          if (!r.method) return;
          var key = [r.dataset, r.mission, r.task, r.n, r.m, r.D, r.ensemble].join('||');
          if (!groups[key]) groups[key] = { times: {}, notSupported: {} };
          if (r.not_supported || r.memory_crash || r.runtime_error) {
            groups[key].notSupported[r.method] = true;
          } else {
            if (!groups[key].times[r.method]) groups[key].times[r.method] = [];
            groups[key].times[r.method].push(r.running_time);
          }
        });
        var totals = {}, counts = {}, nGroups = 0;
        Object.keys(groups).forEach(function(key) {
          var g = groups[key];
          var times = {};
          Object.keys(g.times).forEach(function(m) {
            var avg = g.times[m].reduce(function(a,b){return a+b;},0) / g.times[m].length;
            if (avg > 0) times[m] = avg;
          });
          var nsKeys = Object.keys(g.notSupported);
          // Need at least 2 methods total (supported or crashed) to form a comparison
          if (Object.keys(times).length + nsKeys.length < 2) return;
          if (Object.keys(times).length === 0) {
            // All methods crashed — all get score 0
            nsKeys.forEach(function(m) {
              totals[m] = (totals[m] || 0) + 0;
              counts[m] = (counts[m] || 0) + 1;
            });
            nGroups++;
            return;
          }
          var minT = Math.min.apply(null, Object.keys(times).map(function(m){ return times[m]; }));
          if (removeFast) {
            var allFast = Object.keys(times).every(function(m){ return times[m] < 10; });
            if (allFast) return;
          }
          Object.keys(times).forEach(function(m) {
            var score = (minT / times[m]) * 100;
            totals[m] = (totals[m] || 0) + score;
            counts[m] = (counts[m] || 0) + 1;
          });
          nsKeys.forEach(function(m) {
            totals[m] = (totals[m] || 0) + 0;
            counts[m] = (counts[m] || 0) + 1;
          });
          nGroups++;
        });
        if (!nGroups) return null;
        var scores = {};
        Object.keys(totals).forEach(function(m) { scores[m] = totals[m] / counts[m]; });
        return { scores: scores, n: nGroups };
      }

      function updateFilteredScore() {
        var nLo = sbVals.n[sbState.n.lo], nHi = sbVals.n[sbState.n.hi];
        var mLo = sbVals.m[sbState.m.lo], mHi = sbVals.m[sbState.m.hi];
        var dLo = sbVals.D[sbState.D.lo], dHi = sbVals.D[sbState.D.hi];
        var selTasks = [];
        document.querySelectorAll('.sb-task-cb').forEach(function(cb) {
          if (cb.checked) selTasks.push(cb.value);
        });
        var selMethods = [];
        document.querySelectorAll('.sb-method-cb').forEach(function(cb) {
          if (cb.checked) selMethods.push(cb.value);
        });
        var rmFastEl = document.getElementById('sb-remove-fast');
        var removeFast = rmFastEl ? rmFastEl.checked : false;
        var filtered = DATA.filter(function(r) {
          return r.n >= nLo && r.n <= nHi && r.m >= mLo && r.m <= mHi && r.D >= dLo && r.D <= dHi &&
                 (selTasks.length === 0 || selTasks.indexOf(r.task) !== -1) &&
                 (selMethods.length === 0 || selMethods.indexOf(r.method) !== -1);
        });
        var result = computeScoreFromRows(filtered, removeFast);
        var el2 = document.getElementById('sb-filtered-score');
        if (!result) {
          el2.innerHTML = '<span style="color:#adb5bd;font-size:0.83rem">No comparable runs in range.</span>';
          return;
        }
        var maxScore = Math.max.apply(null, Object.keys(result.scores).map(function(m){ return result.scores[m]; }));
        var html = Object.keys(result.scores).sort().map(function(m) {
          return renderMethodBadge(m, result.scores[m], maxScore);
        }).join('<span class="msb-sep">vs</span>');
        html += '<span style="color:#adb5bd;font-size:0.78rem;margin-left:6px">(' + result.n + ' runs)</span>';
        el2.innerHTML = html;
      }

      SB_PARAMS.forEach(function(p) {
        var loEl = document.getElementById('sb-drs-' + p + '-lo');
        var hiEl = document.getElementById('sb-drs-' + p + '-hi');
        function syncSbZIndex() {
          loEl.style.zIndex = (parseInt(loEl.value) >= parseInt(loEl.max)) ? 3 : '';
        }
        loEl.addEventListener('input', function() {
          if (parseInt(loEl.value) > parseInt(hiEl.value)) loEl.value = hiEl.value;
          syncSbZIndex();
          sbState[p].lo = parseInt(loEl.value);
          sbState[p].hi = parseInt(hiEl.value);
          document.getElementById('sb-drs-' + p + '-lo-val').textContent = sbVals[p][sbState[p].lo];
          document.getElementById('sb-drs-' + p + '-hi-val').textContent = sbVals[p][sbState[p].hi];
          updateSbFill(p);
          updateFilteredScore();
        });
        hiEl.addEventListener('input', function() {
          if (parseInt(hiEl.value) < parseInt(loEl.value)) hiEl.value = loEl.value;
          syncSbZIndex();
          sbState[p].lo = parseInt(loEl.value);
          sbState[p].hi = parseInt(hiEl.value);
          document.getElementById('sb-drs-' + p + '-lo-val').textContent = sbVals[p][sbState[p].lo];
          document.getElementById('sb-drs-' + p + '-hi-val').textContent = sbVals[p][sbState[p].hi];
          updateSbFill(p);
          updateFilteredScore();
        });
        syncSbZIndex();
        updateSbFill(p);
      });

      document.querySelectorAll('.sb-task-cb').forEach(function(cb) {
        cb.addEventListener('change', updateFilteredScore);
      });
      document.querySelectorAll('.sb-method-cb').forEach(function(cb) {
        cb.addEventListener('change', updateFilteredScore);
      });
      var sbRmFastEl = document.getElementById('sb-remove-fast');
      if (sbRmFastEl) sbRmFastEl.addEventListener('change', updateFilteredScore);

      updateFilteredScore();
    }

    function renderMissionScore(missionName) {
      var el = document.getElementById('mission-score-banner');
      var entry = (SCORES.by_mission || {})[missionName];
      if (!entry || !entry.scores) { el.innerHTML = ''; return; }

      var maxScore = Math.max.apply(null, Object.keys(entry.scores).map(function(m){ return entry.scores[m]; }));
      var badges = Object.keys(entry.scores).sort().map(function(m) {
        return renderMethodBadge(m, entry.scores[m], maxScore);
      });

      el.innerHTML =
        '<span class="msb-label">Mission Score:</span>' +
        badges.join('<span class="msb-sep">vs</span>') +
        '<span style="color:#adb5bd;font-size:0.78rem;margin-left:6px">avg of ' +
        entry.n + ' run' + (entry.n > 1 ? 's' : '') + '</span>';
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
    function taskDisplayName(t) {
      if (t === 'background_shap')    return 'Background SHAP';
      if (t === 'path_dependent_shap') return 'Path-Dependent SHAP';
      return t;
    }

    function unique(arr) {
      return [...new Set(arr)].sort(function(a, b) {
        return typeof a === 'number' ? a - b : String(a).localeCompare(String(b));
      });
    }

    function getUnique(key, subset) {
      var src = subset !== undefined ? subset : DATA;
      return unique(src.map(function(r) { return r[key]; }));
    }

    // -----------------------------------------------------------------------
    // State
    // -----------------------------------------------------------------------
    var allDatasets = getUnique('dataset');
    var state = {
      dataset: allDatasets[0],
      mission: null,
    };

    // -----------------------------------------------------------------------
    // Cascading options helpers
    // -----------------------------------------------------------------------
    function rowsForDataset() {
      return DATA.filter(function(r) { return r.dataset === state.dataset; });
    }

    function rowsForMission() {
      return rowsForDataset().filter(function(r) { return r.mission === state.mission; });
    }

    function missionsForDataset() { return getUnique('mission', rowsForDataset()); }

    // -----------------------------------------------------------------------
    // Auto-detect x-axis from the current mission's data
    // -----------------------------------------------------------------------
    var X_LABELS = {
      n: 'n \u2014 explain set size',
      m: 'm \u2014 background size',
      D: 'D \u2014 max tree depth',
    };

    function detectXParam(rows) {
      var ns = getUnique('n', rows);
      var ms = getUnique('m', rows);
      var ds = getUnique('D', rows);
      if (ns.length > 1) return 'n';
      if (ds.length > 1) return 'D';
      if (ms.length > 1) return 'm';
      return 'n';
    }

    // -----------------------------------------------------------------------
    // Chart
    // -----------------------------------------------------------------------
    function buildTraces(rows, xp, yPos) {
      var approaches = unique(rows.map(function(r) { return r.approach; }));
      return approaches.map(function(app) {
        // Include crash rows (at yPos) alongside normal rows so the line extends to them.
        // Exclude not_supported rows — those remain in their own separate trace.
        var appRows = rows
          .filter(function(r) { return r.approach === app && !r.not_supported; })
          .sort(function(a, b) { return a[xp] - b[xp]; });
        if (!appRows.length) return null;

        var label = methodLabel(appRows[0].method);
        var color = methodColor(appRows[0].method);
        return {
          x: appRows.map(function(r) { return r[xp]; }),
          y: appRows.map(function(r) {
            return (r.memory_crash || r.runtime_error) ? yPos : r.running_time;
          }),
          error_y: {
            type: 'data',
            array: appRows.map(function(r) {
              return (r.memory_crash || r.runtime_error || r.is_estimated) ? 0 : r.std_s;
            }),
            visible: true,
          },
          mode: 'lines+markers',
          name: label,
          line: { color: color },
          marker: {
            symbol: appRows.map(function(r) {
              if (r.memory_crash) return 'triangle-down';
              if (r.runtime_error) return 'x';
              if (r.is_estimated) return 'circle-open';
              return 'circle';
            }),
            size: appRows.map(function(r) {
              return (r.memory_crash || r.runtime_error) ? 13 : 9;
            }),
            color: appRows.map(function(r) {
              if (r.memory_crash) return '#d62728';
              if (r.runtime_error) return '#ff7f0e';
              return color;
            }),
          },
          customdata: appRows.map(function(r) {
            if (r.memory_crash) return 'Memory Crash';
            if (r.runtime_error) return 'Runtime Error';
            var t = fmtTime(r.running_time);
            if (r.is_estimated) {
              t += '\u2009\u2605\u202festimated';
              if (r.estimation_description) t += '<br><i>' + r.estimation_description + '</i>';
            }
            return t;
          }),
          hovertemplate:
            '<b>' + label + '</b><br>' +
            xp + ': %{x}<br>' +
            'time: %{customdata}' +
            '<extra></extra>',
        };
      });
    }

    function buildNotSupportedTraces(rows, xp, yPos) {
      var approaches = unique(rows.map(function(r) { return r.approach; }));
      var traces = [];
      approaches.forEach(function(app) {
        var nsRows = rows
          .filter(function(r) { return r.approach === app && r.not_supported; })
          .sort(function(a, b) { return a[xp] - b[xp]; });
        if (!nsRows.length) return;
        var color = methodColor(nsRows[0].method);
        var label = methodLabel(nsRows[0].method) + ' \u2014 not supported';
        traces.push({
          x: nsRows.map(function(r) { return r[xp]; }),
          y: nsRows.map(function() { return yPos; }),
          mode: 'lines+markers',
          name: label,
          line: { dash: 'dash', color: color, width: 1.5 },
          marker: { symbol: 'x', size: 11, color: color, line: { width: 2.5, color: color } },
          hovertemplate: '<b>' + label + '</b><br>' + xp + ': %{x}<extra></extra>',
        });
      });
      return traces;
    }

    function shouldUseLogScale(vals) {
      var nums = vals.filter(function(v) { return v > 0; });
      if (nums.length < 2) return false;
      var mn = Math.min.apply(null, nums);
      var mx = Math.max.apply(null, nums);
      return mx / mn >= 20;
    }

    function renderChart(rows, xp) {
      var xVals = getUnique(xp, rows);
      var supportedRows = rows.filter(function(r) { return !r.not_supported && !r.memory_crash && !r.runtime_error; });
      var yVals = supportedRows.map(function(r) { return r.running_time; }).filter(function(v) { return v > 0; });
      var useLogX = shouldUseLogScale(xVals);
      var useLogY = shouldUseLogScale(yVals);

      var hasNotSupported = rows.some(function(r) { return r.not_supported; });
      var hasMemoryCrash  = rows.some(function(r) { return r.memory_crash; });
      var hasRuntimeError = rows.some(function(r) { return r.runtime_error; });
      var hasSpecial = hasNotSupported || hasMemoryCrash || hasRuntimeError;
      var yMax = yVals.length ? Math.max.apply(null, yVals) : 1;
      var yMin = yVals.length ? Math.min.apply(null, yVals) : 0;
      var yPos, yAxisRange;
      if (useLogY) {
        yPos = yMax * 100;
        yAxisRange = [Math.log10(Math.max(yMin * 0.5, 1e-6)), Math.log10(yPos * 3)];
      } else {
        yPos = yMax * 1.35;
        yAxisRange = hasSpecial ? [0, yMax * 1.7] : null;
      }

      var traces = buildTraces(rows, xp, yPos).filter(function(t) { return t !== null; });
      if (hasNotSupported) {
        traces = traces.concat(buildNotSupportedTraces(rows, xp, yPos));
      }

      var yAxis = { title: 'Time (s)', type: useLogY ? 'log' : 'linear' };
      if (yAxisRange) yAxis.range = yAxisRange;

      var layout = {
        xaxis: { title: X_LABELS[xp] || xp, type: useLogX ? 'log' : 'linear' },
        yaxis: yAxis,
        legend: { title: { text: 'Method\u2003\u25cb=estimated\u2003\u25bc/\u00d7=crash' } },
        template: 'plotly_white',
        margin: { t: 30, r: 20, l: 70, b: 55 },
        height: 460,
      };
      Plotly.react('chart', traces, layout);
    }

    // -----------------------------------------------------------------------
    // Table
    // -----------------------------------------------------------------------
    function renderTable(rows, xp) {
      var el = document.getElementById('data-table');
      if (rows.length === 0) {
        el.innerHTML = '<p>No data matches the current selection.</p>';
        return;
      }

      var xVals     = getUnique(xp, rows);
      var approaches = getUnique('approach', rows);

      // index[approach][xVal] = {t, est, ns, mc, re}
      var index = {};
      rows.forEach(function(r) {
        if (!index[r.approach]) index[r.approach] = {};
        index[r.approach][r[xp]] = { t: r.running_time, est: r.is_estimated, ns: r.not_supported, mc: r.memory_crash, re: r.runtime_error };
      });

      var html = '<table><thead><tr><th>Approach</th>';
      xVals.forEach(function(v) { html += '<th>' + xp + '\u202f=\u202f' + v + '</th>'; });
      html += '</tr></thead><tbody>';

      approaches.forEach(function(app) {
        html += '<tr><td class="app-name">' + app + '</td>';
        xVals.forEach(function(v) {
          var cell = (index[app] || {})[v];
          if (cell !== undefined) {
            if (cell.ns || cell.mc || cell.re) {
              html += '<td class="missing">N/A</td>';
            } else {
              var val = fmtTime(cell.t);
              html += '<td class="' + (cell.est ? 'estimated' : '') + '">'
                    + val + (cell.est ? '*' : '') + '</td>';
            }
          } else {
            html += '<td class="missing">\u2014</td>';
          }
        });
        html += '</tr>';
      });
      html += '</tbody></table>';
      el.innerHTML = html;
    }

    // -----------------------------------------------------------------------
    // Model card helper (called by the depth/variant select box)
    // -----------------------------------------------------------------------
    function updateModelCard(uid) {
      var data = (window._modelRegistry || {})[uid];
      if (!data) return;
      var sel = document.getElementById(uid + '-sel');
      if (!sel) return;
      var item = data[parseInt(sel.value, 10)];
      if (!item) return;
      Object.keys(item.varying).forEach(function(k) {
        var cell = document.getElementById(uid + '-v-' + k);
        if (cell) cell.textContent = item.varying[k];
      });
      var tEl  = document.getElementById(uid + '-T');
      var lEl  = document.getElementById(uid + '-L');
      var tlEl = document.getElementById(uid + '-TL');
      if (tEl)  tEl.textContent  = item.T  != null ? item.T            : '\u2014';
      if (lEl)  lEl.textContent  = item.L  != null ? item.L.toFixed(1) : '\u2014';
      if (tlEl) tlEl.textContent = (item.T != null && item.L != null)
                                     ? Math.round(item.T * item.L) : '\u2014';
    }

    // -----------------------------------------------------------------------
    // Details panel
    // -----------------------------------------------------------------------
    function renderDetails(missionName, rows) {
      var el = document.getElementById('mission-details-content');
      var m = MISSION_META[missionName];
      if (!m) { el.innerHTML = '<p style="color:#888;font-size:0.85rem">No metadata available.</p>'; return; }

      var html = '';

      // --- Dataset card ---
      var ds = m.dataset || {};
      html += '<div class="detail-card">';
      html += '<h3>Dataset</h3>';
      html += '<dl>';
      html += '<dt>Name</dt><dd>' + (ds.name || '\u2014') + '</dd>';
      html += '<dt>Rows</dt><dd>' + (ds.n_samples != null ? ds.n_samples.toLocaleString() : '\u2014') + '</dd>';
      html += '<dt>Features</dt><dd>' + (ds.n_features != null ? ds.n_features : '\u2014') + '</dd>';
      if (ds.columns && ds.columns.length) {
        var shown = ds.columns.slice(0, 10);
        var extra = ds.columns.length - shown.length;
        html += '<dt>Feature names</dt><dd>' + shown.join(', ') + (extra > 0 ? ' \u2026 +' + extra + ' more' : '') + '</dd>';
      }
      html += '</dl>';
      html += '</div>';

      // --- Model(s) card ---
      var models = m.models || [];
      html += '<div class="detail-card">';
      html += '<h3>Models</h3>';

      // Reset registry so stale UIDs from a previous mission don't linger
      window._modelRegistry = {};

      // Group models by ensemble_type so each type gets one card
      var byEnsemble = {}, ensembleOrder = [];
      models.forEach(function(mod) {
        var et = mod.ensemble_type || 'unknown';
        if (!byEnsemble[et]) { byEnsemble[et] = []; ensembleOrder.push(et); }
        byEnsemble[et].push(mod);
      });

      ensembleOrder.forEach(function(et) {
        var mods = byEnsemble[et];
        var allKeys = Object.keys(mods[0].hyperparams || {});

        // Split params into shared (same value across all models) vs varying
        var sharedParams = {}, varyingKeys = [];
        allKeys.forEach(function(k) {
          var vals = mods.map(function(mod) { return String((mod.hyperparams || {})[k]); });
          if (vals.every(function(v) { return v === vals[0]; })) {
            sharedParams[k] = (mods[0].hyperparams || {})[k];
          } else {
            varyingKeys.push(k);
          }
        });

        html += '<dl>';
        html += '<dt>Type</dt><dd>' + et + '</dd>';
        Object.keys(sharedParams).forEach(function(k) {
          html += '<dt>' + k + '</dt><dd>' + sharedParams[k] + '</dd>';
        });

        if (mods.length > 1 && varyingKeys.length > 0) {
          // Multi-model group: show a select for the varying params
          var uid = 'mcard-' + et.replace(/\W/g, '') + '-' + Date.now();

          // Pre-compute T/L for each model by matching rows on D or index
          var modelData = mods.map(function(mod, i) {
            var hp = mod.hyperparams || {};
            var label = varyingKeys.map(function(k) { return k + '=' + hp[k]; }).join(', ');
            var targetD = hp['max_depth'];
            var matchRows = (rows || []).filter(function(r) { return r.ensemble === et; });
            var mr = null;
            if (targetD != null) {
              mr = matchRows.find(function(r) { return r.D === targetD; });
              if (!mr) mr = matchRows.reduce(function(best, r) {
                return (!best || Math.abs(r.D - targetD) < Math.abs(best.D - targetD)) ? r : best;
              }, null);
            } else {
              mr = matchRows[i] || null;
            }
            return {
              label: label,
              varying: varyingKeys.reduce(function(acc, k) { acc[k] = hp[k]; return acc; }, {}),
              T: mr ? mr.T : null,
              L: mr ? mr.L : null,
            };
          });
          window._modelRegistry[uid] = modelData;

          // Varying-param display rows (value updates when select changes)
          varyingKeys.forEach(function(k) {
            html += '<dt>' + k + '</dt><dd id="' + uid + '-v-' + k + '">'
                  + (mods[0].hyperparams || {})[k] + '</dd>';
          });
          html += '<dt>\u25bc model</dt><dd>'
                + '<select id="' + uid + '-sel" '
                + 'onchange="updateModelCard(\'' + uid + '\')" '
                + 'style="font-size:0.8rem;padding:2px 6px;border-radius:4px;border:1px solid #ccc">';
          modelData.forEach(function(md, i) {
            html += '<option value="' + i + '">' + md.label + '</option>';
          });
          html += '</select></dd>';
          html += '<dt>trees (T)</dt><dd id="' + uid + '-T">\u2014</dd>';
          html += '<dt>avg leaves/tree (L)</dt><dd id="' + uid + '-L">\u2014</dd>';
          html += '<dt>total leaves</dt><dd id="' + uid + '-TL">\u2014</dd>';

        } else {
          // Single model — find the one matching row
          var sampleRow = (rows || []).find(function(r) { return r.ensemble === et; });
          if (sampleRow) {
            html += '<dt>trees (T)</dt><dd>' + sampleRow.T + '</dd>';
            html += '<dt>avg leaves/tree (L)</dt><dd>' + sampleRow.L.toFixed(1) + '</dd>';
            html += '<dt>total leaves</dt><dd>' + Math.round(sampleRow.T * sampleRow.L) + '</dd>';
          }
        }

        html += '</dl>';
      });

      html += '</div>';

      // --- Task & approaches card ---
      var tasks = m.tasks || [];
      html += '<div class="detail-card">';
      html += '<h3>Task &amp; Approaches</h3>';
      tasks.forEach(function(t) {
        html += '<p style="font-size:0.85rem;font-weight:600;color:#1f3a5f;margin:6px 0 4px">' + t.name + '</p>';
        (t.approaches || []).forEach(function(a) {
          html += '<div class="approach-row">'
                + '<span class="aname">' + a.name + '</span>'
                + (a.description ? '<span class="adesc">\u2014 ' + a.description + '</span>' : '')
                + '</div>';
        });
      });
      html += '</div>';

      // --- Sweep card ---
      html += '<div class="detail-card">';
      html += '<h3>Mission Sweep</h3>';
      html += '<dl>';
      html += '<dt>n values</dt><dd>' + (m.n_values || []).join(', ') + '</dd>';
      html += '<dt>m values</dt><dd>' + (m.m_values || []).join(', ') + '</dd>';
      html += '</dl>';
      html += '</div>';

      el.innerHTML = html;
      // Populate the model select cards with the first-option values
      Object.keys(window._modelRegistry || {}).forEach(function(uid) {
        updateModelCard(uid);
      });
    }

    // -----------------------------------------------------------------------
    // Main render
    // -----------------------------------------------------------------------
    function render() {
      var rows = rowsForMission();

      renderDetails(state.mission, rows);
      renderMissionScore(state.mission);

      if (rows.length === 0) {
        Plotly.react('chart', [], { title: 'No data for this selection', template: 'plotly_white' });
        document.getElementById('data-table').innerHTML = '<p>No data matches the current selection.</p>';
        return;
      }

      var xp = detectXParam(rows);
      renderChart(rows, xp);
      renderTable(rows, xp);
    }

    // -----------------------------------------------------------------------
    // Select population helpers
    // -----------------------------------------------------------------------
    function populateSelect(id, values, current) {
      var el = document.getElementById(id);
      el.innerHTML = '';
      values.forEach(function(v) {
        var opt = document.createElement('option');
        opt.value = v;
        opt.text  = String(v);
        if (String(v) === String(current)) opt.selected = true;
        el.appendChild(opt);
      });
      // Return the actually-selected value (first if current wasn't in list).
      return el.options[el.selectedIndex] ? el.options[el.selectedIndex].value : (values[0] || null);
    }

    function refreshMissionSelect() {
      var missions = missionsForDataset();
      if (missions.indexOf(state.mission) === -1) state.mission = missions[0] || null;
      populateSelect('ctrl-mission', missions, state.mission);
      state.mission = missions.indexOf(state.mission) !== -1 ? state.mission : (missions[0] || null);
    }

    // -----------------------------------------------------------------------
    // Initialise
    // -----------------------------------------------------------------------
    populateSelect('ctrl-dataset', allDatasets, state.dataset);
    refreshMissionSelect();

    document.getElementById('ctrl-dataset').addEventListener('change', function(e) {
      state.dataset = e.target.value;
      refreshMissionSelect();
      render();
    });
    document.getElementById('ctrl-mission').addEventListener('change', function(e) {
      state.mission = e.target.value;
      render();
    });

    renderScoreboard();
    render();

    // -----------------------------------------------------------------------
    // All-results table — dual-range slider filters
    // -----------------------------------------------------------------------
    var AR_PARAMS  = ['n', 'm', 'D'];

    var arVals  = {};   // param -> sorted unique value array
    var arState = {};   // param -> {lo: idx, hi: idx}

    function buildDualSliders() {
      AR_PARAMS.forEach(function(p) {
        arVals[p]  = getUnique(p);
        arState[p] = { lo: 0, hi: arVals[p].length - 1 };
      });

      var html = '';
      AR_PARAMS.forEach(function(p) {
        var vals = arVals[p];
        var maxI = vals.length - 1;
        html +=
          '<div class="ar-filter-group">' +
          '<label>' + p + ':</label>' +
          '<div class="dual-range-wrap">' +
            '<div class="dual-range-slider">' +
              '<div class="drs-track"></div>' +
              '<div class="drs-fill" id="drs-' + p + '-fill"></div>' +
              '<input type="range" id="drs-' + p + '-lo" min="0" max="' + maxI + '" value="0">' +
              '<input type="range" id="drs-' + p + '-hi" min="0" max="' + maxI + '" value="' + maxI + '">' +
            '</div>' +
            '<div class="drs-vals">' +
              '<span id="drs-' + p + '-lo-val">' + vals[0] + '</span>' +
              '<span id="drs-' + p + '-hi-val">' + vals[maxI] + '</span>' +
            '</div>' +
          '</div>' +
          '</div>';
      });

      var arTaskVals = getUnique('task');
      html += '<div class="ar-filter-group"><span class="fr-label">Tasks:</span><div class="filter-chips">';
      arTaskVals.forEach(function(t) {
        html += '<label class="filter-chip"><input type="checkbox" class="ar-task-cb" value="' + t + '" checked> ' + taskDisplayName(t) + '</label>';
      });
      html += '</div></div>';

      var arDatasetVals = getUnique('dataset');
      html += '<div class="ar-filter-group"><span class="fr-label">Datasets:</span><div class="filter-chips">';
      arDatasetVals.forEach(function(d) {
        html += '<label class="filter-chip"><input type="checkbox" class="ar-dataset-cb" value="' + d + '" checked> ' + d + '</label>';
      });
      html += '</div></div>';

      var arEnsembleVals = getUnique('ensemble');
      html += '<div class="ar-filter-group"><span class="fr-label">Ensembles:</span><div class="filter-chips">';
      arEnsembleVals.forEach(function(e) {
        html += '<label class="filter-chip"><input type="checkbox" class="ar-ensemble-cb" value="' + e + '" checked> ' + e + '</label>';
      });
      html += '</div></div>';

      var allMethodNames = SCORES.methods || [];
      html += '<div class="ar-filter-group"><label style="font-weight:600;color:#495057;white-space:nowrap">Winner:</label>';
      html += '<select id="ar-winner-sel" style="border:1px solid #ced4da;border-radius:4px;padding:3px 7px;font-size:0.84rem;background:#fff;cursor:pointer">';
      html += '<option value="all">All runs</option>';
      allMethodNames.forEach(function(m) {
        html += '<option value="' + m + '">' + methodLabel(m) + ' wins</option>';
      });
      html += '</select></div>';

      html += '<div class="ar-filter-group"><label class="remove-fast-wrap"><input type="checkbox" id="ar-remove-fast"> Runtime &gt; 10s</label></div>';
      document.getElementById('ar-filters').innerHTML = html;

      document.querySelectorAll('.ar-task-cb').forEach(function(cb) {
        cb.addEventListener('change', renderAllResultsTable);
      });
      document.querySelectorAll('.ar-dataset-cb').forEach(function(cb) {
        cb.addEventListener('change', renderAllResultsTable);
      });
      document.querySelectorAll('.ar-ensemble-cb').forEach(function(cb) {
        cb.addEventListener('change', renderAllResultsTable);
      });
      var arWinnerEl = document.getElementById('ar-winner-sel');
      if (arWinnerEl) arWinnerEl.addEventListener('change', renderAllResultsTable);
      var arRmFastEl = document.getElementById('ar-remove-fast');
      if (arRmFastEl) arRmFastEl.addEventListener('change', renderAllResultsTable);

      AR_PARAMS.forEach(function(p) {
        var loEl = document.getElementById('drs-' + p + '-lo');
        var hiEl = document.getElementById('drs-' + p + '-hi');
        function syncZIndex() {
          var lo = parseInt(loEl.value);
          var max = parseInt(loEl.max);
          // Raise lo above hi only when both handles are at the right end.
          // hi sits on top naturally (DOM order), which is correct for every
          // other position — including both-at-zero — so the user can drag it
          // right.  The one exception is both-at-max: there hi would block lo
          // from being dragged left, so we flip the stacking order.
          loEl.style.zIndex = (lo >= max) ? 3 : '';
        }
        loEl.addEventListener('input', function() {
          if (parseInt(loEl.value) > parseInt(hiEl.value)) loEl.value = hiEl.value;
          syncZIndex();
          onSliderChange(p);
        });
        hiEl.addEventListener('input', function() {
          if (parseInt(hiEl.value) < parseInt(loEl.value)) hiEl.value = loEl.value;
          syncZIndex();
          onSliderChange(p);
        });
        syncZIndex();
        updateFill(p);
      });
    }

    function onSliderChange(p) {
      var lo = parseInt(document.getElementById('drs-' + p + '-lo').value);
      var hi = parseInt(document.getElementById('drs-' + p + '-hi').value);
      arState[p] = { lo: lo, hi: hi };
      document.getElementById('drs-' + p + '-lo-val').textContent = arVals[p][lo];
      document.getElementById('drs-' + p + '-hi-val').textContent = arVals[p][hi];
      updateFill(p);
      renderAllResultsTable();
    }

    function updateFill(p) {
      var n = arVals[p].length;
      var fill = document.getElementById('drs-' + p + '-fill');
      if (!fill) return;
      if (n <= 1) { fill.style.left = '0%'; fill.style.width = '100%'; return; }
      var pct = 100 / (n - 1);
      fill.style.left  = (arState[p].lo * pct) + '%';
      fill.style.width = ((arState[p].hi - arState[p].lo) * pct) + '%';
    }

    function renderAllResultsTable() {
      var selArTasks = [];
      document.querySelectorAll('.ar-task-cb').forEach(function(cb) {
        if (cb.checked) selArTasks.push(cb.value);
      });
      var selDatasets = [];
      document.querySelectorAll('.ar-dataset-cb').forEach(function(cb) {
        if (cb.checked) selDatasets.push(cb.value);
      });
      var selEnsembles = [];
      document.querySelectorAll('.ar-ensemble-cb').forEach(function(cb) {
        if (cb.checked) selEnsembles.push(cb.value);
      });
      var arWinnerMethod = document.getElementById('ar-winner-sel') ? document.getElementById('ar-winner-sel').value : 'all';
      var arRmFast = document.getElementById('ar-remove-fast') ? document.getElementById('ar-remove-fast').checked : false;

      // Filter individual rows by param ranges, task, dataset, and ensemble
      var filtered = DATA.filter(function(r) {
        if (!AR_PARAMS.every(function(p) {
          return r[p] >= arVals[p][arState[p].lo] && r[p] <= arVals[p][arState[p].hi];
        })) return false;
        if (selArTasks.length > 0 && selArTasks.indexOf(r.task) === -1) return false;
        if (selDatasets.length > 0 && selDatasets.indexOf(r.dataset) === -1) return false;
        if (selEnsembles.length > 0 && selEnsembles.indexOf(r.ensemble) === -1) return false;
        return true;
      });

      // Group by run: one row per (dataset, mission, task, n, m, D, ensemble)
      var runKeys = [];
      var runMap = {};
      filtered.forEach(function(r) {
        var k = [r.dataset, r.mission, r.task, r.n, r.m, r.D, r.ensemble].join('||');
        if (!runMap[k]) {
          runMap[k] = { dataset: r.dataset, mission: r.mission, task: r.task,
                        n: r.n, m: r.m, D: r.D, T: r.T, L: r.L, F: r.F,
                        ensemble: r.ensemble, methods: {} };
          runKeys.push(k);
        }
        runMap[k].methods[r.method] = { t: r.running_time, est: r.is_estimated,
                                         ns: r.not_supported, mc: r.memory_crash, re: r.runtime_error };
      });

      // Apply winner filter: keep only runs where arWinnerMethod has the lowest valid time
      if (arWinnerMethod !== 'all') {
        runKeys = runKeys.filter(function(k) {
          var mths = runMap[k].methods;
          var winnerCell = mths[arWinnerMethod];
          if (!winnerCell || winnerCell.ns || winnerCell.mc || winnerCell.re) return false;
          var winnerTime = winnerCell.t;
          return Object.keys(mths).every(function(m) {
            if (m === arWinnerMethod) return true;
            var c = mths[m];
            if (c.ns || c.mc || c.re) return true; // crashed methods don't count against winner
            return winnerTime <= c.t;
          });
        });
      }

      // Apply remove-fast filter at run level: skip runs where all valid times < 10 s
      if (arRmFast) {
        runKeys = runKeys.filter(function(k) {
          var mths = runMap[k].methods;
          return !Object.keys(mths).every(function(m) {
            var c = mths[m];
            return c.ns || c.mc || c.re || c.t < 10;
          });
        });
      }

      // Determine methods present in the (filtered) runs, sorted
      var methodSet = {};
      runKeys.forEach(function(k) {
        Object.keys(runMap[k].methods).forEach(function(m) { methodSet[m] = true; });
      });
      var methods = Object.keys(methodSet).sort();

      var BASE_COLS = ['#', 'Dataset', 'Task', 'Mission', 'n', 'm', 'D', 'T', 'L', 'F', 'Ensemble'];
      var html = '<table><thead><tr>';
      BASE_COLS.forEach(function(h) { html += '<th>' + h + '</th>'; });
      methods.forEach(function(m) { html += '<th>' + methodLabel(m) + '</th>'; });
      html += '</tr></thead><tbody>';

      runKeys.forEach(function(k, idx) {
        var run = runMap[k];
        html += '<tr>';
        html += '<td>' + (idx + 1) + '</td>';
        html += '<td>' + run.dataset + '</td>';
        html += '<td>' + run.task + '</td>';
        html += '<td>' + run.mission + '</td>';
        html += '<td>' + run.n + '</td>';
        html += '<td>' + run.m + '</td>';
        html += '<td>' + run.D + '</td>';
        html += '<td>' + run.T + '</td>';
        html += '<td>' + (run.L != null ? run.L.toFixed(1) : '\u2014') + '</td>';
        html += '<td>' + run.F + '</td>';
        html += '<td>' + run.ensemble + '</td>';
        methods.forEach(function(m) {
          var cell = run.methods[m];
          if (!cell) {
            html += '<td class="missing">\u2014</td>';
          } else if (cell.ns || cell.mc || cell.re) {
            html += '<td class="missing">N/A</td>';
          } else {
            html += '<td class="time-cell' + (cell.est ? ' estimated' : '') + '">'
                  + fmtTime(cell.t) + (cell.est ? '*' : '') + '</td>';
          }
        });
        html += '</tr>';
      });
      html += '</tbody></table>';
      document.getElementById('all-results-table').innerHTML = html;
    }

    buildDualSliders();
    renderAllResultsTable();

    // -----------------------------------------------------------------------
    // Analytics section
    // -----------------------------------------------------------------------
    var _anaVals = {}, _anaState = {};
    var _ANA_PARAMS = ['n', 'm', 'D'];

    function _anaUpdateFill(p) {
      var n = _anaVals[p].length;
      var fill = document.getElementById('ana-drs-' + p + '-fill');
      if (!fill) return;
      if (n <= 1) { fill.style.left = '0%'; fill.style.width = '100%'; return; }
      var pct = 100 / (n - 1);
      fill.style.left  = (_anaState[p].lo * pct) + '%';
      fill.style.width = ((_anaState[p].hi - _anaState[p].lo) * pct) + '%';
    }

    function buildAnaFilters() {
      _ANA_PARAMS.forEach(function(p) {
        _anaVals[p]  = getUnique(p);
        _anaState[p] = { lo: 0, hi: _anaVals[p].length - 1 };
      });

      var html = '';

      // n / m / D dual-range sliders
      _ANA_PARAMS.forEach(function(p) {
        var vals = _anaVals[p];
        var maxI = vals.length - 1;
        html +=
          '<div class="ar-filter-group">' +
          '<label>' + p + ':</label>' +
          '<div class="dual-range-wrap">' +
            '<div class="dual-range-slider">' +
              '<div class="drs-track"></div>' +
              '<div class="drs-fill" id="ana-drs-' + p + '-fill"></div>' +
              '<input type="range" id="ana-drs-' + p + '-lo" min="0" max="' + maxI + '" value="0">' +
              '<input type="range" id="ana-drs-' + p + '-hi" min="0" max="' + maxI + '" value="' + maxI + '">' +
            '</div>' +
            '<div class="drs-vals">' +
              '<span id="ana-drs-' + p + '-lo-val">' + vals[0] + '</span>' +
              '<span id="ana-drs-' + p + '-hi-val">' + vals[maxI] + '</span>' +
            '</div>' +
          '</div></div>';
      });

      // Task chips
      var taskVals = getUnique('task');
      html += '<div class="ar-filter-group"><span class="fr-label">Tasks:</span><div class="filter-chips">';
      taskVals.forEach(function(t) {
        html += '<label class="filter-chip"><input type="checkbox" class="ana-task-cb" value="' + t + '" checked> ' + taskDisplayName(t) + '</label>';
      });
      html += '</div></div>';

      // Dataset chips
      var dsVals = getUnique('dataset');
      html += '<div class="ar-filter-group"><span class="fr-label">Datasets:</span><div class="filter-chips">';
      dsVals.forEach(function(d) {
        html += '<label class="filter-chip"><input type="checkbox" class="ana-dataset-cb" value="' + d + '" checked> ' + d + '</label>';
      });
      html += '</div></div>';

      // Winner dropdown
      var allMethods = SCORES.methods || [];
      html += '<div class="ar-filter-group"><label style="font-weight:600;color:#495057;white-space:nowrap">Winner:</label>';
      html += '<select id="ana-winner-sel" style="border:1px solid #ced4da;border-radius:4px;padding:3px 7px;font-size:0.84rem;background:#fff;cursor:pointer">';
      html += '<option value="all">All runs</option>';
      allMethods.forEach(function(m) {
        html += '<option value="' + m + '">' + methodLabel(m) + ' wins</option>';
      });
      html += '</select></div>';

      document.getElementById('ana-filters').innerHTML = html;

      // Wire sliders
      _ANA_PARAMS.forEach(function(p) {
        var loEl = document.getElementById('ana-drs-' + p + '-lo');
        var hiEl = document.getElementById('ana-drs-' + p + '-hi');
        function syncZ() { loEl.style.zIndex = (parseInt(loEl.value) >= parseInt(loEl.max)) ? 3 : ''; }
        loEl.addEventListener('input', function() {
          if (parseInt(loEl.value) > parseInt(hiEl.value)) loEl.value = hiEl.value;
          syncZ();
          _anaState[p].lo = parseInt(loEl.value);
          _anaState[p].hi = parseInt(hiEl.value);
          document.getElementById('ana-drs-' + p + '-lo-val').textContent = _anaVals[p][_anaState[p].lo];
          document.getElementById('ana-drs-' + p + '-hi-val').textContent = _anaVals[p][_anaState[p].hi];
          _anaUpdateFill(p);
          renderAnalytics();
        });
        hiEl.addEventListener('input', function() {
          if (parseInt(hiEl.value) < parseInt(loEl.value)) hiEl.value = loEl.value;
          syncZ();
          _anaState[p].lo = parseInt(loEl.value);
          _anaState[p].hi = parseInt(hiEl.value);
          document.getElementById('ana-drs-' + p + '-lo-val').textContent = _anaVals[p][_anaState[p].lo];
          document.getElementById('ana-drs-' + p + '-hi-val').textContent = _anaVals[p][_anaState[p].hi];
          _anaUpdateFill(p);
          renderAnalytics();
        });
        syncZ();
        _anaUpdateFill(p);
      });

      document.querySelectorAll('.ana-task-cb').forEach(function(cb) {
        cb.addEventListener('change', renderAnalytics);
      });
      document.querySelectorAll('.ana-dataset-cb').forEach(function(cb) {
        cb.addEventListener('change', renderAnalytics);
      });
      var anaWinEl = document.getElementById('ana-winner-sel');
      if (anaWinEl) anaWinEl.addEventListener('change', renderAnalytics);
    }

    function getAnaFilteredRuns() {
      var selTasks = [];
      document.querySelectorAll('.ana-task-cb').forEach(function(cb) {
        if (cb.checked) selTasks.push(cb.value);
      });
      var selDatasets = [];
      document.querySelectorAll('.ana-dataset-cb').forEach(function(cb) {
        if (cb.checked) selDatasets.push(cb.value);
      });
      var winnerFilter = (document.getElementById('ana-winner-sel') || {}).value || 'all';

      // Filter individual rows by all active filters
      var filtered = DATA.filter(function(r) {
        if (!_ANA_PARAMS.every(function(p) {
          return r[p] >= _anaVals[p][_anaState[p].lo] && r[p] <= _anaVals[p][_anaState[p].hi];
        })) return false;
        if (selTasks.length > 0 && selTasks.indexOf(r.task) === -1) return false;
        if (selDatasets.length > 0 && selDatasets.indexOf(r.dataset) === -1) return false;
        return true;
      });

      // Group into runs (unique dataset × mission × task × n × m × D × ensemble)
      var runMap = {}, runKeys = [];
      filtered.forEach(function(r) {
        var k = [r.dataset, r.mission, r.task, r.n, r.m, r.D, r.ensemble].join('||');
        if (!runMap[k]) {
          runMap[k] = {
            dataset: r.dataset, mission: r.mission, task: r.task,
            n: r.n, m: r.m, D: r.D, ensemble: r.ensemble,
            methods: {}
          };
          runKeys.push(k);
        }
        runMap[k].methods[r.method] = {
          t: r.running_time,
          ns: r.not_supported,
          mc: r.memory_crash,
          re: r.runtime_error
        };
      });

      // Compute winner per run (method with lowest positive valid time)
      runKeys.forEach(function(k) {
        var g = runMap[k];
        var best = null, bestT = Infinity;
        Object.keys(g.methods).forEach(function(m) {
          var c = g.methods[m];
          if (!c.ns && !c.mc && !c.re && c.t > 0 && c.t < bestT) {
            bestT = c.t; best = m;
          }
        });
        g.winner = best || '__all_failed__';
      });

      // Apply winner filter
      if (winnerFilter !== 'all') {
        runKeys = runKeys.filter(function(k) { return runMap[k].winner === winnerFilter; });
      }

      return { runMap: runMap, runKeys: runKeys };
    }

    function _makePieLayout(title, color) {
      return {
        title: { text: title, font: { size: 13, color: color || '#343a40' } },
        margin: { t: 44, b: 8, l: 8, r: 8 },
        height: 270,
        showlegend: true,
        legend: { font: { size: 10 }, orientation: 'v' },
        paper_bgcolor: 'rgba(0,0,0,0)',
      };
    }

    function renderAnaMissionsPies(runMap, runKeys) {
      var el = document.getElementById('ana-missions-pies');
      if (runKeys.length === 0) {
        el.innerHTML = '<p style="color:#888;font-size:0.9rem;padding:8px 0">No runs match the current filters.</p>';
        return;
      }

      // Single row: n, m, D (numeric), then Task type, Ensemble type (categorical).
      // Visibility is based on ALL runs in DATA, not the current filter selection.
      var allDims = [
        { key: 'n',        label: 'n — explain set size', numeric: true },
        { key: 'm',        label: 'm — background size',  numeric: true,
          skip: _allUniqueM.length === 1 && _allUniqueM[0] === '0' },
        { key: 'D',        label: 'D — max tree depth',   numeric: true },
        { key: 'task',     label: 'Task type',            fmt: taskDisplayName,
          skip: _allUniqueTask.length <= 1 },
        { key: 'ensemble', label: 'Ensemble type',
          skip: _allUniqueEnsemble.length <= 1 },
      ].filter(function(d) { return !d.skip; });

      var piesHtml = '<div class="ana-pies-row">';
      allDims.forEach(function(d) {
        var cls = d.fmt || !d.numeric ? 'ana-pie-cell-wide' : 'ana-pie-cell';
        piesHtml += '<div id="ana-pie-' + d.key + '" class="' + cls + '"></div>';
      });
      piesHtml += '</div>';
      el.innerHTML = piesHtml;

      allDims.forEach(function(dim) {
        var counts = {};
        runKeys.forEach(function(k) {
          var val = String(runMap[k][dim.key]);
          if (dim.fmt) val = dim.fmt(val);
          counts[val] = (counts[val] || 0) + 1;
        });
        var labels = Object.keys(counts).sort(function(a, b) {
          if (dim.numeric) return parseFloat(a) - parseFloat(b);
          return String(a).localeCompare(String(b));
        });
        var values = labels.map(function(l) { return counts[l]; });
        var trace = { type: 'pie', labels: labels, values: values, hole: 0.35,
          textinfo: 'percent',
          hovertemplate: '<b>%{label}</b><br>%{value} runs (%{percent})<extra></extra>' };
        if (dim.numeric) { trace.sort = false; trace.direction = 'clockwise'; }
        Plotly.newPlot('ana-pie-' + dim.key, [trace], _makePieLayout(dim.label),
          { displayModeBar: false, responsive: true });
      });
    }

    function renderAnaScore(runMap, runKeys) {
      var el = document.getElementById('ana-score');
      if (!el) return;
      if (runKeys.length === 0) {
        el.innerHTML = '<span style="color:#adb5bd;font-size:0.83rem">No comparable runs in range.</span>';
        return;
      }

      // Compute scores from already-grouped runs
      var totals = {}, counts = {}, nGroups = 0;
      runKeys.forEach(function(k) {
        var g = runMap[k];
        var times = {};
        Object.keys(g.methods).forEach(function(m) {
          var c = g.methods[m];
          if (!c.ns && !c.mc && !c.re && c.t > 0) times[m] = c.t;
        });
        var nsKeys = Object.keys(g.methods).filter(function(m) {
          var c = g.methods[m]; return c.ns || c.mc || c.re;
        });
        if (Object.keys(times).length + nsKeys.length < 2) return;
        if (Object.keys(times).length === 0) {
          nsKeys.forEach(function(m) { totals[m] = (totals[m] || 0); counts[m] = (counts[m] || 0) + 1; });
          nGroups++;
          return;
        }
        var minT = Math.min.apply(null, Object.keys(times).map(function(m) { return times[m]; }));
        Object.keys(times).forEach(function(m) {
          totals[m] = (totals[m] || 0) + (minT / times[m]) * 100;
          counts[m] = (counts[m] || 0) + 1;
        });
        nsKeys.forEach(function(m) { totals[m] = (totals[m] || 0); counts[m] = (counts[m] || 0) + 1; });
        nGroups++;
      });

      if (!nGroups) {
        el.innerHTML = '<span style="color:#adb5bd;font-size:0.83rem">No comparable runs in range.</span>';
        return;
      }
      var scores = {};
      Object.keys(totals).forEach(function(m) { scores[m] = totals[m] / counts[m]; });
      var maxScore = Math.max.apply(null, Object.keys(scores).map(function(m) { return scores[m]; }));
      var html = Object.keys(scores).sort().map(function(m) {
        return renderMethodBadge(m, scores[m], maxScore);
      }).join('<span class="msb-sep">vs</span>');
      html += '<span style="color:#adb5bd;font-size:0.78rem;margin-left:6px">(' + nGroups + ' runs)</span>';
      el.innerHTML = html;
    }

    function renderAnaMethodsPies(runMap, runKeys) {
      var el = document.getElementById('ana-methods-pies');
      var methods = SCORES.methods || [];
      if (runKeys.length === 0 || methods.length === 0) { el.innerHTML = ''; return; }

      var piesHtml = '<div class="ana-pies-row">';
      piesHtml += '<div id="ana-pie-winner" class="ana-pie-cell-wide"></div>';
      methods.forEach(function(m) {
        var safeId = m.replace(/[^a-zA-Z0-9]/g, '_');
        piesHtml += '<div id="ana-mpie-' + safeId + '" class="ana-pie-cell"></div>';
      });
      piesHtml += '</div>';
      el.innerHTML = piesHtml;

      // Winner pie
      var wCounts = {};
      runKeys.forEach(function(k) {
        var w = runMap[k].winner;
        var lbl = (w === '__all_failed__') ? 'All Failed' : methodLabel(w);
        wCounts[lbl] = (wCounts[lbl] || 0) + 1;
      });
      var wLabels = Object.keys(wCounts);
      var wValues = wLabels.map(function(l) { return wCounts[l]; });
      var wColors = wLabels.map(function(l) {
        if (l === 'All Failed') return '#d62728';
        var mn = (SCORES.methods || []).find(function(m) { return methodLabel(m) === l; });
        return mn ? methodColor(mn) : '#adb5bd';
      });
      Plotly.newPlot('ana-pie-winner',
        [{ type: 'pie', labels: wLabels, values: wValues, hole: 0.35,
           marker: { colors: wColors }, textinfo: 'percent',
           hovertemplate: '<b>%{label}</b><br>%{value} runs (%{percent})<extra></extra>' }],
        _makePieLayout('Winner'),
        { displayModeBar: false, responsive: true }
      );

      var STATUS_LABELS = ['Won', 'Lost', 'Not Supported', 'Memory Crash', 'Exception Crash'];
      var STATUS_KEYS   = ['won', 'lost', 'not_supported', 'memory_crash', 'runtime_error'];
      var STATUS_COLORS = ['#2ca02c', '#1f77b4', '#adb5bd', '#d62728', '#ff7f0e'];

      methods.forEach(function(m) {
        var counts = { won: 0, lost: 0, not_supported: 0, memory_crash: 0, runtime_error: 0 };
        runKeys.forEach(function(k) {
          var g = runMap[k];
          var cell = g.methods[m];
          if (!cell) return;
          if (cell.ns)        { counts.not_supported++; }
          else if (cell.mc)   { counts.memory_crash++;  }
          else if (cell.re)   { counts.runtime_error++; }
          else if (g.winner === m) { counts.won++; }
          else                { counts.lost++;          }
        });

        var filtLabels = [], filtValues = [], filtColors = [];
        STATUS_KEYS.forEach(function(sk, i) {
          if (counts[sk] > 0) {
            filtLabels.push(STATUS_LABELS[i]);
            filtValues.push(counts[sk]);
            filtColors.push(STATUS_COLORS[i]);
          }
        });
        if (filtValues.length === 0) {
          filtLabels = ['No data']; filtValues = [1]; filtColors = ['#dee2e6'];
        }

        var safeId = m.replace(/[^a-zA-Z0-9]/g, '_');
        Plotly.newPlot(
          'ana-mpie-' + safeId,
          [{ type: 'pie', labels: filtLabels, values: filtValues, hole: 0.35,
             marker: { colors: filtColors }, textinfo: 'percent',
             hovertemplate: '<b>%{label}</b><br>%{value} runs (%{percent})<extra></extra>' }],
          _makePieLayout(methodLabel(m), methodColor(m)),
          { displayModeBar: false, responsive: true }
        );
      });
    }

    function renderAnalytics() {
      var result = getAnaFilteredRuns();
      renderAnaScore(result.runMap, result.runKeys);
      renderAnaMissionsPies(result.runMap, result.runKeys);
      renderAnaMethodsPies(result.runMap, result.runKeys);
    }

    buildAnaFilters();
    renderAnalytics();

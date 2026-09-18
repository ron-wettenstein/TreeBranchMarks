    // =======================================================================
    // Canonical scoring — the single source of truth for every score shown.
    //
    // This is a faithful port of html_generator._compute_scores().  Python
    // computes the SCORES constant once (over all rows); this function
    // recomputes the same thing in the browser over an arbitrary subset of
    // rows, which is what the global Methods filter needs.  The two MUST agree
    // exactly when given the same rows — tests/verification depend on it, so
    // keep the two implementations in lockstep when either changes.
    //
    //   computeScores(rows, opts) -> {
    //     methods:    [approach, ...]            sorted
    //     overall:    { scores: {approach: avg}, n: nGroups } | null
    //     by_mission: { missionName: { scores, n } }
    //   }
    //
    // opts.removeFast — drop groups where every supported time is < 10 s.
    //   Used only by the Filtered Score box; Python has no equivalent.
    // =======================================================================
    function computeScores(rows, opts) {
      var removeFast = !!(opts && opts.removeFast);

      // --- Group rows -------------------------------------------------------
      // T is part of the key so a mission that sweeps ensemble size scores each
      // size separately instead of averaging them into one group.  This mirrors
      // the Python key exactly; leaving T out makes the filtered score disagree
      // with the score summary on T-sweeping missions.
      var groups = {}, groupKeys = [];
      rows.forEach(function(r) {
        if (!r.approach) return;
        var key = [r.dataset, r.mission, r.task, r.n, r.m, r.D, r.T, r.ensemble].join('||');
        var g = groups[key];
        if (!g) {
          g = groups[key] = { times: {}, notSupported: {}, mission: '', task: '' };
          groupKeys.push(key);
        }
        if (r.not_supported || r.memory_crash || r.runtime_error) {
          g.notSupported[r.approach] = true;
        } else {
          if (!g.times[r.approach]) g.times[r.approach] = [];
          g.times[r.approach].push(r.running_time);
        }
        g.mission = r.mission;
        g.task    = r.task;
      });

      // --- Score each group -------------------------------------------------
      var runs = [];
      var allMethods = {};

      groupKeys.forEach(function(key) {
        var g = groups[key];

        // Mean time per approach (only approaches that recorded at least one time).
        var timesByMethod = {}, timeKeys = [];
        Object.keys(g.times).forEach(function(a) {
          var ts = g.times[a];
          if (!ts.length) return;
          var sum = ts.reduce(function(x, y) { return x + y; }, 0);
          timesByMethod[a] = sum / ts.length;
          timeKeys.push(a);
        });

        var supported = {}, supportedKeys = [];
        timeKeys.forEach(function(a) {
          if (timesByMethod[a] > 0) { supported[a] = timesByMethod[a]; supportedKeys.push(a); }
        });

        var nsKeys = Object.keys(g.notSupported);

        // Need at least 2 approaches total (timed or crashed) to form a
        // comparison.  Note this counts timeKeys, NOT supportedKeys: an
        // approach that recorded a non-crash time of 0 still counts toward the
        // group size even though it gets no score.  Python does the same.
        var inGroup = {};
        timeKeys.forEach(function(a) { inGroup[a] = true; });
        nsKeys.forEach(function(a) { inGroup[a] = true; });
        if (Object.keys(inGroup).length < 2) return;

        var scores = {};
        if (supportedKeys.length) {
          var winnerTime = Infinity;
          supportedKeys.forEach(function(a) {
            if (supported[a] < winnerTime) winnerTime = supported[a];
          });
          if (removeFast && supportedKeys.every(function(a) { return supported[a] < 10; })) {
            return;
          }
          supportedKeys.forEach(function(a) {
            scores[a] = (winnerTime / supported[a]) * 100.0;
          });
        }
        nsKeys.forEach(function(a) { scores[a] = 0.0; });

        Object.keys(scores).forEach(function(a) { allMethods[a] = true; });
        runs.push({ mission: g.mission, task: g.task, scores: scores });
      });

      // --- Average over a set of runs ---------------------------------------
      function avgScores(subset) {
        if (!subset.length) return null;
        var totals = {}, counts = {};
        subset.forEach(function(run) {
          Object.keys(run.scores).forEach(function(a) {
            totals[a] = (totals[a] || 0.0) + run.scores[a];
            counts[a] = (counts[a] || 0) + 1;
          });
        });
        var out = {};
        Object.keys(totals).forEach(function(a) { out[a] = totals[a] / counts[a]; });
        return { scores: out, n: subset.length };
      }

      var byMission = {};
      var missionNames = {};
      runs.forEach(function(run) { missionNames[run.mission] = true; });
      Object.keys(missionNames).forEach(function(mn) {
        byMission[mn] = avgScores(runs.filter(function(run) { return run.mission === mn; }));
      });

      return {
        methods: Object.keys(allMethods).sort(),
        overall: avgScores(runs),
        by_mission: byMission,
      };
    }

window.initCgFeatureDetail = async function({visState, renderAll, data, cgSel}){
  var sel = cgSel.select('.feature-detail').html('')
  if (!sel.node()) return

  var headerSel = sel.append('div.feature-header')
  var logitsSel = sel.append('div.logits-container')
  var examplesSel = sel.append('div.feature-examples-container')
  var featureExamples = await window.initFeatureExamples({
    containerSel: examplesSel,
    showLogits: true,
    // showLogits: !data.nodes.some(d => d.top_logit_effects) // we show logits ourselves frozen above the feature vis, don't also show it inside
  })

  let editOpen = false;

  // throttle to prevent lag when mousing over
  var renderFeatureExamples = util.throttleDebounce(featureExamples.renderFeature, 200)

  function renderFeatureDetail() {
    logitsSel.html('').st({display:''})

    // display hovered then clicked nodes, with fallbacks for supernode
    var d = data.nodes.find(e => e.nodeId === visState.hoveredNodeId)
    if (!d) d = data.nodes.find(e => e.nodeId === visState.clickedId)
    if (!d){
      var featureId = visState.hoveredId
      if (!featureId || featureId.includes('supernode')){
        headerSel.html('')
          .append('div.no-selected-feature').text("Click or hover to see a feature's examples")
        examplesSel.st({opacity: 0})
        return
      }
      return
    }

    if (visState.graphSchemaVersion > 0) {
      var [layerIdx, featIdx] = util.cantorUnpair(d.featureIndex)
    } else {
      var featIdx = d.featureIndex
    }

    var label = d.isTmpFeature ? d.featureId :
      visState.isHideLayer ? `#F${d.featureIndex}` :
      `${utilCg.layerLocationLabel(d.layer, d.probe_location_idx)}/${featIdx}`

    if (d.isError || d.feature_type == 'embedding' || d.feature_type == 'logit'){
      if (d.isError) addLogits(d)
      if (d.feature_type=='logit') addEmbeddings(d)

      headerSel.html('').append('div.header-top-row').append('div.feature-title')
        .text(d.ppClerp)
      examplesSel.st({opacity: 0})
    } else if (d.feature_type == 'cross layer transcoder') {
      const scan = data.metadata.scan?.startsWith('custom-') ? data.metadata.transcoder_list[d.layer] : data.metadata.scan;
      addLogits(d)
      addEmbeddings(d)
      var headerTopRowSel = headerSel.html('').append('div.header-top-row')

      const currentActivation = d.activation

      const actText = typeof currentActivation == 'number' ? currentActivation.toFixed(2) : 'N/A'
      const featureTitleSel = headerTopRowSel.append('div.feature-title')
        .html(`Feature&nbsp;<a style="color: inherit;" href="${d.url}" target="_blank">${label}</a> <span style="font-size: 0.9em; color: #777;">Act: ${actText}</span>`)

      if (typeof currentActivation == 'number' && scan && scan.includes('//')) {
        window.renderActHistogram({
          featureTitleSel,
          scan,
          featureNode: d,
          featureExamples,
        })
      }

      headerTopRowSel.append('div.pp-clerp')
        .text(d.ppClerp)
        .at({title: d.ppClerp})

      if (visState.isEditMode){
        headerTopRowSel.append('button.edit-clerp-button')
          .text('Edit')
          .on('click', toggleEdit)

        function toggleEdit() {
          editOpen = !editOpen;
          hClerpEditSel.st({display: editOpen ? 'flex' : 'none'})
          if (editOpen) {
            headerSel.select('input').node()?.focus();
          }
        }

        const hClerpEditSel = headerSel.append('div.clerp-edit')
          .st({ display: editOpen ? 'flex' : 'none' });

        const hClerpSel = hClerpEditSel.append('div')
          .st({ display: 'flex' });
        hClerpSel.append('div')
          .st({flex: '0 0 50px'})
          .text(`🧑💻`);
        hClerpSel.append('input').data([d])
          .at({ value: d.localClerp })
          .st({flex: '1 0 0', whiteSpace: 'normal', fontSize: 12})
          .on('change', ev => {
            renderAll.hClerpUpdate([d, ev.target.value]);
          })

        // const rClerpSel = hClerpEditSel.append('div')
        //     .st({ display: 'flex' });
        //   rClerpSel.append('div')
        //     .st({flex: '0 0 50px'})
        //     .text(`🧑☁️`);
        //   rClerpSel.append('div')
        //     .text(d.remoteClerp)
        //     .st({flex: '1 0', whiteSpace: 'normal'})

        // const clerpSel = hClerpEditSel.append('div')
        //   .st({ display: 'flex' });
        // clerpSel.append('div')
        //   .st({flex: '0 0 50px'})
        //   .text(`🤖💬`);
        // clerpSel.append('div')
        //   .text(d.clerp)
        //   .st({ flex: '1 0', whiteSpace: 'normal' })
      }

      // Show cached exemplars immediately, otherwise show a load button
      examplesSel.st({opacity: 1})
      examplesSel.selectAll('.load-exemplars-btn').remove()
      var sign = (typeof d.activation == 'number' && d.activation < 0) ? '-' : '+'
      if (featureExamples.isCached(d.featureIndex, sign)) {
        renderFeatureExamples(scan, d.featureIndex, undefined, sign)
      } else {
        featureExamples.clearExamples()
        examplesSel.insert('button', ':first-child')
          .attr('class', 'load-exemplars-btn')
          .text('Load exemplars')
          .st({margin: '8px', padding: '4px 12px', cursor: 'pointer'})
          .on('click', function() {
            d3.select(this).text('Loading...').attr('disabled', true)
            var btn = d3.select(this)
            renderFeatureExamples(scan, d.featureIndex, undefined, sign).then(() => btn.remove())
          })
      }
    } else {
      headerSel.html(`<b>${label}</b>`)
      logitsSel.html('No logit data')
      examplesSel.st({opacity: 0})
    }

    // add pinned/click state and toggle to feature-title
    headerSel.select('div.feature-title')
      .classed('pinned', d.nodeId && visState.pinnedIds.includes(d.nodeId))
      .classed('hovered', visState.clickedId == d.nodeId)
      .on('click', ev => {
        utilCg.clickFeature(visState, renderAll, d, ev.metaKey || ev.ctrlKey)

        if (visState.clickedId) return
        // double render to toggle on hoveredId, could expose more of utilCg.clickFeature to prevent
        visState.hoveredId = d.featureId
        renderAll.hoveredId()
      })

  }

  function addLogits(d) {
    // return
    if (!d || !d.top_logit_effects) return logitsSel.html('').st({display: 'none'})
    // Add logit effects section
    let logitRowContainerSel = logitsSel.st({display: ''})
      .append('div.effects')
      .appendMany('div.sign', [d.top_logit_effects, d.bottom_logit_effects].filter(d => d))
    logitRowContainerSel.append('div.label').text((d, i) => i ? 'Bottom Outputs' : 'Top Outputs')
    logitRowContainerSel.append('div.rows')
      .appendMany('div.row', d => d)
      .append('span.key').text(d => d)
  }
  function addEmbeddings(d) {
    // return
    // Add embedding effects section
    if (d.top_embedding_effects || d.bottom_embedding_effects) {
      let embeddingRowContainerSel = logitsSel
        .append('div.effects')
        .appendMany('div.sign', [d.top_embedding_effects, d.bottom_embedding_effects].filter(d => d))
      embeddingRowContainerSel.append('div.label').text((d, i) => i ? 'Bottom Inputs' : 'Top Inputs')
      embeddingRowContainerSel.append('div.rows').appendMany('div.row', d => d)
        .append('span.key').text(d => d)
    }
  }

  function addAttrContribMaps(d) {
    if (!d.attr_map && !d.contrib_map) return

    var mapsSel = logitsSel.append('div.attr-contrib-maps')
      .st({marginTop: 6, fontSize: 12})

    if (d.attr_map) {
      var tokens = data.metadata.prompt_tokens || []
      // Align attr_map to suffix of prompt_tokens when lengths differ
      var offset = tokens.length - d.attr_map.length
      var section = mapsSel.append('div').st({marginBottom: 8})
      section.append('div').st({fontWeight: 'bold', marginBottom: 2}).text('Input Attribution (attr_map)')
      var bars = section.append('div').st({display: 'flex', flexWrap: 'wrap', gap: '1px'})
      var maxVal = Math.max(...d.attr_map.map(Math.abs), 1e-8)
      d.attr_map.forEach((v, i) => {
        var tokenIdx = i + Math.max(0, offset)
        var frac = v / maxVal
        var color = frac >= 0
          ? `rgba(37, 99, 235, ${Math.abs(frac).toFixed(2)})`
          : `rgba(220, 38, 38, ${Math.abs(frac).toFixed(2)})`
        bars.append('div')
          .st({
            padding: '1px 3px',
            background: color,
            color: Math.abs(frac) > 0.5 ? '#fff' : '#333',
            borderRadius: 2,
            whiteSpace: 'nowrap',
            cursor: 'default',
          })
          .text(tokens[tokenIdx] || `[${i}]`)
          .at({title: `${tokens[tokenIdx] || i}: ${v.toFixed(4)}`})
      })
    }

    if (d.contrib_map) {
      var logitTokens = data.metadata.target_logit_tokens || []
      var section = mapsSel.append('div').st({marginBottom: 8})
      section.append('div').st({fontWeight: 'bold', marginBottom: 2}).text('Output Contribution (contrib_map)')
      var bars = section.append('div').st({display: 'flex', flexWrap: 'wrap', gap: '1px'})
      var maxVal = Math.max(...d.contrib_map.map(Math.abs), 1e-8)
      d.contrib_map.forEach((v, i) => {
        var frac = v / maxVal
        var color = frac >= 0
          ? `rgba(37, 99, 235, ${Math.abs(frac).toFixed(2)})`
          : `rgba(220, 38, 38, ${Math.abs(frac).toFixed(2)})`
        bars.append('div')
          .st({
            padding: '1px 3px',
            background: color,
            color: Math.abs(frac) > 0.5 ? '#fff' : '#333',
            borderRadius: 2,
            whiteSpace: 'nowrap',
            cursor: 'default',
          })
          .text(logitTokens[i] || `logit[${i}]`)
          .at({title: `${logitTokens[i] || i}: ${v.toFixed(4)}`})
      })
    }
  }

  renderAll.hClerpUpdate.fns.push(renderFeatureDetail)
  renderAll.clickedId.fns.push(renderFeatureDetail)
  renderAll.hoveredId.fns.push(renderFeatureDetail)
  renderAll.pinnedIds.fns.push(renderFeatureDetail)

  renderFeatureDetail()
}

window.init?.()

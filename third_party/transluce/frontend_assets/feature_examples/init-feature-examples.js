window.initFeatureExamples = function({containerSel, showLogits=true, showExamples=true, hideStaleOutputs=false}){
  var visState = {
    isDev: 0,
    showLogits,
    showExamples,
    hideStaleOutputs,

    activeToken: null,
    feature: null,
    featureIndex: -1,

    chartRowTop: 16,
    chartRowHeight: 82,
  }

  var indexFileExistsCache = new Map()
  var neuronAPICache = new Map()

  // set up dom and render fns
  var sel = containerSel.html('').append('div.feature-examples')
  if (visState.showLogits) sel.append('div.feature-example-logits')
  if (visState.showExamples) sel.append('div.feature-example-list')
  var renderAll = util.initRenderAll(['feature'])


  if (visState.showLogits) window.initFeatureExamplesLogits({renderAll, visState, sel})
  if (visState.showExamples) window.initFeatureExamplesList({renderAll, visState, sel})

  function isCached(featureIndex, sign) {
    var cacheKey = `${featureIndex}_${sign || '+'}`
    return neuronAPICache.has(cacheKey)
  }

  function clearExamples() {
    visState.featureIndex = -1
    visState.feature = null
    renderAll.feature()
  }

  return {loadFeature, renderFeature, isCached, clearExamples}

  async function renderFeature(scan, featureIndex, layerIdx, sign){
    if (featureIndex == visState.featureIndex) return
    // Clear stale content immediately while loading
    visState.featureIndex = featureIndex
    visState.feature = null
    renderAll.feature()
    if (visState.hideStaleOutputs) sel.classed('is-stale-output', 1)
    var feature = await loadFeature(scan, featureIndex, layerIdx, sign)
    if (feature.featureIndex == visState.featureIndex){
      visState.feature = feature
      renderAll.feature()
      if (visState.hideStaleOutputs) sel.classed('is-stale-output', 0)
    }

    return feature
  }

  function hfUrl(scan, path) {
    const [repoId, rest] = scan.split('//')
    const [filePath, revision] = rest ? rest.split('@') : [null, scan.split('@')[1]]
    const prefix = filePath ? `${filePath}/` : ''
    return `https://huggingface.co/${repoId.split('@')[0]}/resolve/${revision || 'main'}/${prefix}features/${path}`
  }

  function indexFileExists(scan) {
    if (indexFileExistsCache.has(scan)) return indexFileExistsCache.get(scan)

    const promise = fetch(hfUrl(scan, 'index.json.gz'), { method: 'HEAD' })
      .then(response => response.ok)
      .catch(error => {
        if (error.status === 404) {
          return false
        } else {
          throw error
        }
      })

    indexFileExistsCache.set(scan, promise)
    return promise
  }

  async function loadFeatureFromBinary(scan, featureIndex) {
    const [layerIdx, featIdx] = util.cantorUnpair(featureIndex)
    const indexData = await util.getFile(hfUrl(scan, 'index.json.gz'))
    const offsets = indexData[layerIdx]['offsets']
    const binFilename = indexData[layerIdx]['filename']
    const startByte = offsets[featIdx]
    const endByte = offsets[featIdx + 1]

    if (!binFilename || !offsets) {
      throw new Error(`Feature ${featureIndex} not found in index`)
    }

    return await util.getFile(hfUrl(scan, binFilename), true, 'bin', `bytes=${startByte}-${endByte}`)
  }


  async function loadFeatureFromNeuronAPI(featureIndex, sign) {
    const cacheKey = `${featureIndex}_${sign}`
    if (neuronAPICache.has(cacheKey)) return neuronAPICache.get(cacheKey)
    const [layerIdx, neuronIdx] = util.cantorUnpair(featureIndex)
    const signParam = sign === '-' ? '-' : '%2B'
    const response = await fetch(`/api/neuron_exemplars?layer=${layerIdx}&neuron=${neuronIdx}&sign=${signParam}`)
    if (!response.ok) throw new Error(`API returned ${response.status}`)
    const data = await response.json()
    neuronAPICache.set(cacheKey, data)
    return data
  }

  async function loadFeature(scan, featureIndex, layerIdx, sign){
    if (scan.startsWith('./')) {
      var feature = await  util.getFile(`${scan}/${featureIndex}.json`)
    } else if (await indexFileExists(scan)){
      var feature = await loadFeatureFromBinary(scan, featureIndex)
    } else {
      try {
        var feature = await  util.getFile(`./features/${scan}/${featureIndex}.json`)
      } catch(e) {
        var feature = await loadFeatureFromNeuronAPI(featureIndex, sign)
      }
    }

    if (feature.act_min === undefined) {
      feature.act_min = 0
      feature.act_max = 1.4
    }

    feature.colorScale = d3.scaleSequential(d3.interpolateOranges)
      .domain([feature.act_min, feature.act_max]).clamp(1)

    feature.featureIndex = featureIndex
    feature.scan = scan

    return feature
  }
}

window.init?.()

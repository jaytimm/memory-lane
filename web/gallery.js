(() => {
  "use strict";

  const data = window.FAMILY_PHOTO_DATA;
  // Albums show 100 photos at a time. Every search shows 50: the best matches
  // for its words, or a random selection when there are no words.
  const ALBUM_PAGE_SIZE = 100;
  const SEARCH_LIMIT = 50;
  const albums = new Map(data.albums.map((album) => [album.id, album]));
  const photosById = new Map(data.photos.map((photo) => [photo.id, photo]));
  const elements = {
    summary: document.querySelector("#collection-summary"),
    hero: document.querySelector("#hero-strip"),
    controls: document.querySelector("#browse-controls"),
    semanticForm: document.querySelector("#semantic-search-form"),
    semanticField: document.querySelector("#semantic-field"),
    searchExamples: document.querySelector("#search-examples"),
    semanticQuery: document.querySelector("#semantic-query"),
    semanticSubmit: document.querySelector("#semantic-submit"),
    semanticStatus: document.querySelector("#semantic-status"),
    searchExampleGroups: document.querySelector("#search-example-groups"),
    decadeFilter: document.querySelector("#decade-filter"),
    clear: document.querySelector("#clear-filters"),
    albumList: document.querySelector("#album-list"),
    resultsSection: document.querySelector("#photo-results"),
    albumLabel: document.querySelector("#active-album-label"),
    photosHeading: document.querySelector("#photos-heading"),
    period: document.querySelector("#active-period"),
    description: document.querySelector("#active-description"),
    reshuffle: document.querySelector("#reshuffle"),
    count: document.querySelector("#result-count"),
    grid: document.querySelector("#photo-grid"),
    empty: document.querySelector("#empty-state"),
    loadMore: document.querySelector("#load-more"),
    backToSearch: document.querySelector("#back-to-search"),
    guide: document.querySelector("#guide"),
    openGuide: document.querySelector("#open-guide"),
    closeGuide: document.querySelector("#close-guide"),
    viewer: document.querySelector("#viewer"),
    viewerImage: document.querySelector("#viewer-image"),
    faceLayer: document.querySelector("#face-layer"),
    viewerStatus: document.querySelector("#face-status"),
    openOriginal: document.querySelector("#open-original"),
    findSimilar: document.querySelector("#find-similar"),
    findFaces: document.querySelector("#find-faces"),
    viewerTitle: document.querySelector("#viewer-title"),
    viewerAlbum: document.querySelector("#viewer-album"),
  };

  let filteredPhotos = data.photos;
  let shown = ALBUM_PAGE_SIZE;
  let viewerIndex = -1;
  let activeAlbumId = "";
  let readyText = "";
  // Decades used by the last random selection, for "Show different photographs".
  let randomDecades = [];
  let searchPhotos = null;
  let searchMode = null;
  let searchQuery = "";
  let searchDescription = "";
  let imageSearchAvailable = false;
  let faceSearchAvailable = false;
  let originalsAvailable = false;
  // "server" uses the local search service; "browser" uses precomputed
  // vectors when the gallery is opened directly from the USB.
  let searchSource = "server";
  let similarIndex = null;
  let faceIndex = null;
  let textEncoder = null;
  let originalImageUrl = (photo) => `/api/original/${encodeURIComponent(photo.id)}`;
  let originalPageUrl = (photo) => {
    const page = new URL("original-view.html", window.location.href);
    page.searchParams.set("id", photo.id);
    page.searchParams.set("name", photo.filename);
    page.searchParams.set("view", "2");
    return page.href;
  };

  function plural(number, word) {
    return `${number.toLocaleString()} ${word}${number === 1 ? "" : "s"}`;
  }

  function photoDecade(photo) {
    return photo.date_start ? Math.floor(photo.date_start / 10) * 10 : null;
  }

  function decadeLabel(decades) {
    const names = decades.map((decade) => `${decade}s`);
    return names.length > 1 ? `${names.slice(0, -1).join(", ")} and ${names.at(-1)}` : names[0];
  }

  function decadeBoxes() {
    return [...elements.decadeFilter.querySelectorAll("input[data-decade]")];
  }

  function checkedDecades() {
    return decadeBoxes().filter((box) => box.checked).map((box) => Number(box.dataset.decade));
  }

  // Every decade starts ticked; "All years" ticks or unticks them together.
  function updateAllYears() {
    const allYears = elements.decadeFilter.querySelector("input[data-all-years]");
    const ticked = checkedDecades().length;
    allYears.checked = ticked === decadeBoxes().length;
    allYears.indeterminate = ticked > 0 && !allYears.checked;
  }

  function resetDecades() {
    decadeBoxes().forEach((box) => { box.checked = true; });
    updateAllYears();
  }

  function clearSearch(clearInput = false) {
    searchPhotos = null;
    searchMode = null;
    searchQuery = "";
    searchDescription = "";
    if (clearInput) elements.semanticQuery.value = "";
    elements.semanticStatus.textContent = readyText;
    elements.semanticStatus.classList.remove("error");
  }

  // "Clear search" resets everything and returns to the search box.
  function clearEverything() {
    clearSearch(true);
    resetDecades();
    activeAlbumId = "";
    shown = ALBUM_PAGE_SIZE;
    render();
    elements.controls.scrollIntoView({ behavior: "smooth" });
    if (!elements.semanticField.hidden) elements.semanticQuery.focus({ preventScroll: true });
  }

  function initializeFilters() {
    elements.summary.textContent = `${plural(data.photos.length, "photograph")} in ${plural(data.albums.length, "album")}`;
    const heroCount = Math.min(5, data.photos.length);
    for (let index = 0; index < heroCount; index += 1) {
      const photo = data.photos[Math.floor(index * data.photos.length / heroCount)];
      const image = document.createElement("img");
      image.src = photo.thumbnail;
      image.alt = "";
      elements.hero.append(image);
    }

    // Ticking a decade does nothing until Search is pressed.
    const decadeOption = (text, attribute, value) => {
      const option = document.createElement("label");
      option.className = "decade-option";
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = true;
      box.dataset[attribute] = value;
      const name = document.createElement("span");
      name.textContent = text;
      option.append(box, name);
      elements.decadeFilter.append(option);
      return box;
    };
    const allYears = decadeOption("All years", "allYears", "");
    allYears.addEventListener("change", () => {
      decadeBoxes().forEach((box) => { box.checked = allYears.checked; });
      updateAllYears();
    });
    const decades = [...new Set(data.photos.map(photoDecade).filter(Boolean))].sort();
    decades.forEach((decade) => {
      decadeOption(`${decade}s`, "decade", String(decade)).addEventListener("change", updateAllYears);
    });

    data.albums.forEach((album) => {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "album-card";
      card.dataset.albumId = album.id;
      const albumPhotos = data.photos.filter((photo) => photo.album_id === album.id);
      const coverPhoto = albumPhotos[Math.floor(albumPhotos.length / 2)];
      const cover = document.createElement("img");
      cover.src = coverPhoto.thumbnail;
      cover.alt = "";
      cover.loading = "lazy";
      const body = document.createElement("span");
      body.className = "album-card-body";
      const title = document.createElement("strong");
      title.textContent = album.title;
      const details = document.createElement("span");
      details.className = "album-card-count";
      details.textContent = plural(album.actual_count, "photo");
      const note = document.createElement("span");
      note.className = "album-card-note";
      note.textContent = album.description ? album.description.replaceAll(";", " · ") : "Open this album";
      body.append(title, details, note);
      card.append(cover, body);
      card.addEventListener("click", () => {
        clearSearch(true);
        resetDecades();
        activeAlbumId = album.id;
        shown = ALBUM_PAGE_SIZE;
        render();
        elements.photosHeading.scrollIntoView({ behavior: "smooth" });
      });
      elements.albumList.append(card);
    });
  }

  function initializeSearchExamples() {
    const groups = window.MEMORY_LANE_SEARCH_EXAMPLES || [];
    groups.forEach((group) => {
      const section = document.createElement("section");
      section.className = "search-example-group";
      const title = document.createElement("strong");
      title.textContent = group.title;
      const buttons = document.createElement("div");
      buttons.className = "search-example-buttons";
      group.queries.forEach((query) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "search-example-button";
        button.textContent = query;
        // An example only fills the box; Search runs it, like everything else.
        button.addEventListener("click", () => {
          elements.semanticQuery.value = query;
          elements.semanticQuery.focus({ preventScroll: true });
          elements.semanticField.scrollIntoView({ behavior: "smooth", block: "center" });
          elements.semanticStatus.classList.remove("error");
          elements.semanticStatus.textContent = `Click Search to look for “${query}”.`;
        });
        buttons.append(button);
      });
      section.append(title, buttons);
      elements.searchExampleGroups.append(section);
    });
  }

  function applyFilters() {
    if (searchPhotos !== null) {
      filteredPhotos = searchPhotos;
      return;
    }
    filteredPhotos = activeAlbumId ? data.photos.filter((photo) => photo.album_id === activeAlbumId) : [];
  }

  function renderHeading() {
    const album = albums.get(activeAlbumId);
    if (searchMode === "text") {
      elements.albumLabel.textContent = "Search results";
      elements.photosHeading.textContent = `“${searchQuery}”`;
      elements.description.textContent = searchDescription;
    } else if (searchMode === "face") {
      elements.albumLabel.textContent = "Search results";
      elements.photosHeading.textContent = "Photographs with similar faces";
      elements.description.textContent = searchDescription;
    } else if (searchMode === "random") {
      elements.albumLabel.textContent = "Search results";
      elements.photosHeading.textContent = "A random selection";
      elements.description.textContent = searchDescription;
    } else if (searchMode === "similar") {
      elements.albumLabel.textContent = "Search results";
      elements.photosHeading.textContent = "Similar photographs";
      elements.description.textContent = searchDescription;
    } else if (album) {
      elements.albumLabel.textContent = "Album";
      elements.photosHeading.textContent = album.title;
      elements.description.textContent = album.description;
    } else {
      elements.albumLabel.textContent = "All albums";
      elements.photosHeading.textContent = "Photographs";
      elements.description.textContent = "";
    }
  }

  // Search results list the decades their photographs come from.
  function renderPeriod() {
    const decades = searchPhotos === null
      ? []
      : [...new Set(filteredPhotos.map(photoDecade).filter(Boolean))].sort();
    elements.period.hidden = !decades.length;
    elements.period.textContent = decades.length ? `From the ${decadeLabel(decades)}` : "";
  }

  function render() {
    applyFilters();
    elements.grid.replaceChildren();
    renderHeading();
    renderPeriod();
    elements.resultsSection.classList.toggle("search-results", searchPhotos !== null);
    // With nothing chosen there is no photo grid. Only albums are paged.
    const nothingChosen = searchPhotos === null && !activeAlbumId;
    elements.resultsSection.hidden = nothingChosen;
    const limit = nothingChosen ? 0 : (searchPhotos === null ? shown : filteredPhotos.length);
    elements.count.textContent = plural(filteredPhotos.length, "photograph");
    elements.empty.hidden = filteredPhotos.length !== 0;
    elements.loadMore.hidden = limit >= filteredPhotos.length;
    elements.backToSearch.hidden = nothingChosen;
    elements.reshuffle.hidden = searchMode !== "random";

    document.querySelectorAll(".album-card").forEach((card) => {
      card.classList.toggle("active", card.dataset.albumId === activeAlbumId);
    });

    filteredPhotos.slice(0, limit).forEach((photo, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "photo-card";
      button.title = photo.filename;
      const image = document.createElement("img");
      image.src = photo.thumbnail;
      image.alt = photo.filename;
      image.loading = "lazy";
      image.decoding = "async";
      if (photo.width && photo.height && photo.height > photo.width) button.classList.add("portrait");
      button.append(image);
      button.addEventListener("click", () => openViewer(index));
      elements.grid.append(button);
    });
  }

  function openViewer(index) {
    viewerIndex = index;
    const photo = filteredPhotos[index];
    const album = albums.get(photo.album_id);
    elements.viewerImage.onerror = () => {
      elements.viewerImage.onerror = null;
      elements.viewerImage.src = photo.thumbnail;
    };
    // Large photos have a lighter viewing copy; "Go to original" still opens the full file.
    elements.viewerImage.src = photo.viewing || (originalsAvailable ? originalImageUrl(photo) : photo.thumbnail);
    elements.viewerImage.alt = photo.filename;
    elements.viewerTitle.textContent = photo.filename;
    elements.viewerAlbum.textContent = album.title;
    if (originalsAvailable) elements.openOriginal.href = originalPageUrl(photo);
    elements.openOriginal.hidden = !originalsAvailable;
    elements.faceLayer.replaceChildren();
    elements.viewerStatus.textContent = "";
    elements.findSimilar.disabled = false;
    elements.findFaces.disabled = false;
    elements.findSimilar.hidden = !imageSearchAvailable;
    elements.findFaces.hidden = !faceSearchAvailable;
    if (!elements.viewer.open) elements.viewer.showModal();
  }

  function showSearchResults(results, mode, query, description) {
    searchPhotos = results.map((result) => photosById.get(result.id)).filter(Boolean);
    searchMode = mode;
    searchQuery = query || "";
    searchDescription = description;
    render();
    elements.photosHeading.scrollIntoView({ behavior: "smooth" });
  }

  function showFaceCount(faceCount) {
    elements.summary.textContent = `${plural(data.photos.length, "photograph")} · ${plural(data.albums.length, "album")} · ${plural(faceCount, "face")}`;
  }

  async function postJson(path, body, failure) {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || failure);
    return payload;
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = src;
      script.onload = resolve;
      script.onerror = () => reject(new Error("Search data could not be loaded"));
      document.head.append(script);
    });
  }

  // Large vector files load on first use, then stay in memory.
  function loadBrowserIndex(entry, variable, create) {
    return loadScript(`assets/${entry.file}`).then(() => {
      const index = create(window[variable]);
      delete window[variable];
      return index;
    });
  }

  function getSimilarIndex() {
    similarIndex ||= loadBrowserIndex(
      window.MEMORY_LANE_SEARCH_DATA.similar,
      "MEMORY_LANE_SIMILAR_VECTORS",
      window.MemoryLaneBrowserSearch.createSimilarIndex,
    ).catch((error) => { similarIndex = null; throw error; });
    return similarIndex;
  }

  function getFaceIndex() {
    faceIndex ||= loadBrowserIndex(
      window.MEMORY_LANE_SEARCH_DATA.faces,
      "MEMORY_LANE_FACE_VECTORS",
      window.MemoryLaneBrowserSearch.createFaceIndex,
    ).catch((error) => { faceIndex = null; throw error; });
    return faceIndex;
  }

  function getTextEncoder() {
    textEncoder ||= loadBrowserIndex(
      window.MEMORY_LANE_SEARCH_DATA.text,
      "MEMORY_LANE_TEXT_MODEL",
      window.MemoryLaneTextEncoder.createTextEncoder,
    ).catch((error) => { textEncoder = null; throw error; });
    return textEncoder;
  }

  // A photo belongs to the decade its album starts in, as in the decade view.
  function inDecades(photo, decades) {
    return Boolean(photo) && (!decades.length || decades.includes(photoDecade(photo)));
  }

  async function requestTextSearch(query, decades) {
    if (searchSource === "browser") {
      const [encoder, index] = await Promise.all([getTextEncoder(), getSimilarIndex()]);
      // Let the "Searching" message paint before the encoder holds the page.
      await new Promise((resolve) => requestAnimationFrame(() => setTimeout(resolve, 0)));
      const vector = encoder.encode(query);
      return index.search(vector, SEARCH_LIMIT, (id) => inDecades(photosById.get(id), decades));
    }
    // The server takes one year range; results outside the ticked decades are dropped.
    const payload = await postJson("/api/search", {
      query,
      top_k: SEARCH_LIMIT,
      album: null,
      date_from: decades.length ? Math.min(...decades) : null,
      date_to: decades.length ? Math.max(...decades) + 9 : null,
    }, "Search failed");
    return payload.results.filter((result) => inDecades(photosById.get(result.id), decades));
  }

  async function requestSimilar(photoId) {
    if (searchSource === "browser") return (await getSimilarIndex()).similar(photoId, SEARCH_LIMIT);
    const payload = await postJson("/api/similar", { photo_id: photoId, top_k: SEARCH_LIMIT }, "Similar-photo search failed");
    return payload.results;
  }

  async function requestFaces(photoId) {
    if (searchSource === "browser") return (await getFaceIndex()).facesForPhoto(photoId);
    const payload = await postJson("/api/faces", { photo_id: photoId }, "Could not load faces");
    return payload.faces;
  }

  async function requestFaceSearch(faceId) {
    if (searchSource === "browser") {
      return (await getFaceIndex()).search(faceId, SEARCH_LIMIT, photosById);
    }
    const payload = await postJson("/api/face-search", { face_id: faceId, top_k: SEARCH_LIMIT }, "Face search failed");
    return payload.results;
  }

  function imageLoads(src) {
    return new Promise((resolve) => {
      const image = new Image();
      image.onload = () => resolve(true);
      image.onerror = () => resolve(false);
      image.src = src;
    });
  }

  async function enableBrowserOriginals() {
    if (!data.photos.length) return;
    // On the USB, the gallery sits in "Memory Lane/Gallery" beside
    // "Family Photos". A locally built archive may hold its own photos/ copy.
    const locations = [
      (photo) => `../../Family%20Photos/${photo.original.replace(/^photos\//, "")}`,
      (photo) => photo.original,
    ];
    for (const location of locations) {
      if (await imageLoads(location(data.photos[0]))) {
        originalImageUrl = location;
        originalPageUrl = location;
        originalsAvailable = true;
        return;
      }
    }
  }

  function enableBrowserSearch() {
    searchSource = "browser";
    const searchData = window.MEMORY_LANE_SEARCH_DATA;
    if (window.MemoryLaneBrowserSearch && searchData) {
      imageSearchAvailable = Boolean(searchData.similar);
      faceSearchAvailable = Boolean(searchData.faces);
      if (searchData.faces) showFaceCount(searchData.faces.count);
      if (searchData.text && searchData.similar && window.MemoryLaneTextEncoder) {
        enableBrowserTextSearch(searchData.similar.count);
      }
    }
    enableBrowserOriginals();
  }

  // The text model is large, so start loading it as soon as the page is up
  // rather than making the first search wait for it.
  function enableBrowserTextSearch(photoCount) {
    const preparing = "Preparing search… this takes a few seconds.";
    showTypedSearch();
    elements.semanticStatus.textContent = preparing;
    setTimeout(() => {
      Promise.all([getTextEncoder(), getSimilarIndex()]).then(() => {
        readyText = `Ready to search ${photoCount.toLocaleString()} photographs.`;
        if (elements.semanticStatus.textContent === preparing) {
          elements.semanticStatus.textContent = readyText;
        }
      }, () => {
        elements.semanticStatus.textContent = "Typed search could not be prepared on this computer.";
        elements.semanticStatus.classList.add("error");
      });
    }, 300);
  }

  async function enableServerSearch() {
    // Pages opened from file:// cannot reach the search service.
    if (window.location.protocol === "file:") return false;
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (!response.ok) return false;
      const status = await response.json();
      if (!status.available) return false;
      imageSearchAvailable = true;
      faceSearchAvailable = Boolean(status.face_search);
      originalsAvailable = Boolean(status.originals_available);
      showTypedSearch();
      if (status.face_search) showFaceCount(status.face_count);
      readyText = `Ready to search ${status.photo_count.toLocaleString()} photographs.`;
      elements.semanticStatus.textContent = readyText;
      return true;
    } catch (_) {
      return false;
    }
  }

  function showTypedSearch() {
    elements.semanticField.hidden = false;
    elements.searchExamples.hidden = false;
  }

  async function enableSemanticSearch() {
    if (!(await enableServerSearch())) enableBrowserSearch();
  }

  function showRandom(decades) {
    const candidates = data.photos.filter((photo) => inDecades(photo, decades));
    for (let index = candidates.length - 1; index > 0; index -= 1) {
      const swap = Math.floor(Math.random() * (index + 1));
      [candidates[index], candidates[swap]] = [candidates[swap], candidates[index]];
    }
    const sample = candidates.slice(0, SEARCH_LIMIT);
    randomDecades = decades;
    showSearchResults(sample, "random", "", "");
    const within = decades.length ? ` from the ${decadeLabel(decades)}` : "";
    elements.semanticStatus.textContent = `${plural(sample.length, "random photograph")}${within}.`;
  }

  elements.reshuffle.addEventListener("click", () => showRandom(randomDecades));

  elements.semanticForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    // Search applies everything in the box: ticked decades and any words.
    const query = elements.semanticField.hidden ? "" : elements.semanticQuery.value.trim();
    const ticked = checkedDecades();
    elements.semanticStatus.classList.remove("error");
    if (!ticked.length) {
      elements.semanticStatus.textContent = "Tick at least one decade, then click Search.";
      return;
    }
    // All decades ticked means no year filter, which also keeps undated photos.
    const decades = ticked.length === decadeBoxes().length ? [] : ticked;
    const within = decades.length ? ` from the ${decadeLabel(decades)}` : "";
    activeAlbumId = "";
    if (!query) {
      showRandom(decades);
      return;
    }
    elements.semanticSubmit.disabled = true;
    elements.semanticStatus.textContent = "Searching photographs…";
    try {
      const results = await requestTextSearch(query, decades);
      showSearchResults(results, "text", query, "Ranked by visual similarity.");
      elements.semanticStatus.textContent = `${plural(results.length, "result")}${within}, ranked by visual similarity.`;
    } catch (error) {
      elements.semanticStatus.textContent = error.message || "Search failed";
      elements.semanticStatus.classList.add("error");
    } finally {
      elements.semanticSubmit.disabled = false;
    }
  });

  elements.findSimilar.addEventListener("click", async () => {
    const photo = filteredPhotos[viewerIndex];
    if (!photo) return;
    elements.findSimilar.disabled = true;
    elements.findFaces.disabled = true;
    elements.viewerStatus.textContent = "Finding similar photographs…";
    try {
      const results = await requestSimilar(photo.id);
      activeAlbumId = "";
      resetDecades();
      elements.viewer.close();
      showSearchResults(results, "similar", "", "Ranked by whole-photograph similarity.");
    } catch (error) {
      elements.viewerStatus.textContent = error.message || "Similar-photo search failed";
      elements.findSimilar.disabled = false;
      elements.findFaces.disabled = false;
    }
  });

  async function searchForFace(faceId) {
    elements.findSimilar.disabled = true;
    elements.findFaces.disabled = true;
    elements.faceLayer.replaceChildren();
    elements.viewerStatus.textContent = "Finding similar faces…";
    try {
      const results = await requestFaceSearch(faceId);
      activeAlbumId = "";
      resetDecades();
      elements.viewer.close();
      showSearchResults(results, "face", "", "Ranked by face similarity.");
    } catch (error) {
      elements.viewerStatus.textContent = error.message || "Face search failed";
      elements.findSimilar.disabled = false;
      elements.findFaces.disabled = false;
    }
  }

  elements.findFaces.addEventListener("click", async () => {
    const photo = filteredPhotos[viewerIndex];
    if (!photo) return;
    elements.findSimilar.disabled = true;
    elements.findFaces.disabled = true;
    elements.faceLayer.replaceChildren();
    elements.viewerStatus.textContent = "Looking for faces…";
    try {
      const faces = await requestFaces(photo.id);
      if (!faces.length) {
        elements.viewerStatus.textContent = "No clear face was found in this photograph.";
        elements.findSimilar.disabled = false;
        elements.findFaces.disabled = false;
        return;
      }
      if (faces.length === 1) {
        await searchForFace(faces[0].id);
        return;
      }
      faces.forEach((face, index) => {
        const [x, y, width, height] = face.bbox;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "face-choice";
        button.setAttribute("aria-label", `Find faces similar to face ${index + 1}`);
        button.style.left = `${x * 100}%`;
        button.style.top = `${y * 100}%`;
        button.style.width = `${width * 100}%`;
        button.style.height = `${height * 100}%`;
        button.addEventListener("click", () => searchForFace(face.id));
        elements.faceLayer.append(button);
      });
      elements.viewerStatus.textContent = "Choose a face in the photograph.";
      elements.findSimilar.disabled = false;
      elements.findFaces.disabled = false;
    } catch (error) {
      elements.viewerStatus.textContent = error.message || "Could not load faces";
      elements.findSimilar.disabled = false;
      elements.findFaces.disabled = false;
    }
  });

  elements.clear.addEventListener("click", clearEverything);
  elements.loadMore.addEventListener("click", () => { shown += ALBUM_PAGE_SIZE; render(); });
  elements.backToSearch.addEventListener("click", () => elements.controls.scrollIntoView({ behavior: "smooth" }));
  elements.openGuide.addEventListener("click", () => elements.guide.showModal());
  elements.closeGuide.addEventListener("click", () => elements.guide.close());
  elements.guide.addEventListener("click", (event) => { if (event.target === elements.guide) elements.guide.close(); });
  document.querySelector("#close-viewer").addEventListener("click", () => elements.viewer.close());
  document.querySelector("#previous-photo").addEventListener("click", () => openViewer((viewerIndex - 1 + filteredPhotos.length) % filteredPhotos.length));
  document.querySelector("#next-photo").addEventListener("click", () => openViewer((viewerIndex + 1) % filteredPhotos.length));
  elements.viewer.addEventListener("click", (event) => { if (event.target === elements.viewer) elements.viewer.close(); });
  document.addEventListener("keydown", (event) => {
    if (!elements.viewer.open || !filteredPhotos.length) return;
    if (event.key === "ArrowLeft") openViewer((viewerIndex - 1 + filteredPhotos.length) % filteredPhotos.length);
    if (event.key === "ArrowRight") openViewer((viewerIndex + 1) % filteredPhotos.length);
  });

  initializeFilters();
  initializeSearchExamples();
  render();
  enableSemanticSearch();
})();

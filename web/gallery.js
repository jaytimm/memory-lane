(() => {
  "use strict";

  const data = window.FAMILY_PHOTO_DATA;
  const PAGE_SIZE = 80;
  const albums = new Map(data.albums.map((album) => [album.id, album]));
  const photosById = new Map(data.photos.map((photo) => [photo.id, photo]));
  const elements = {
    summary: document.querySelector("#collection-summary"),
    hero: document.querySelector("#hero-strip"),
    controls: document.querySelector("#browse-controls"),
    semanticForm: document.querySelector("#semantic-search-form"),
    semanticQuery: document.querySelector("#semantic-query"),
    semanticSubmit: document.querySelector("#semantic-submit"),
    semanticStatus: document.querySelector("#semantic-status"),
    searchExampleGroups: document.querySelector("#search-example-groups"),
    decadeFilter: document.querySelector("#decade-filter"),
    clear: document.querySelector("#clear-filters"),
    albumList: document.querySelector("#album-list"),
    resultsSection: document.querySelector("#photo-results"),
    resultState: document.querySelector("#result-state"),
    resultStateText: document.querySelector("#result-state-text"),
    clearResultState: document.querySelector("#clear-result-state"),
    albumLabel: document.querySelector("#active-album-label"),
    photosHeading: document.querySelector("#photos-heading"),
    description: document.querySelector("#active-description"),
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
    findSimilar: document.querySelector("#find-similar"),
    findPerson: document.querySelector("#find-person"),
    viewerTitle: document.querySelector("#viewer-title"),
    viewerAlbum: document.querySelector("#viewer-album"),
    original: document.querySelector("#open-original"),
  };

  let filteredPhotos = data.photos;
  let shown = PAGE_SIZE;
  let viewerIndex = -1;
  let activeAlbumId = "";
  let activeDecade = null;
  let searchPhotos = null;
  let searchMode = null;
  let searchQuery = "";
  let searchDescription = "";
  let imageSearchAvailable = false;
  let faceSearchAvailable = false;

  function plural(number, word) {
    return `${number.toLocaleString()} ${word}${number === 1 ? "" : "s"}`;
  }

  function clearSearch(clearInput = false) {
    searchPhotos = null;
    searchMode = null;
    searchQuery = "";
    searchDescription = "";
    if (clearInput) elements.semanticQuery.value = "";
    elements.semanticStatus.textContent = "Search looks at the photographs themselves.";
    elements.semanticStatus.classList.remove("error");
  }

  function clearEverything() {
    clearSearch(true);
    activeAlbumId = "";
    activeDecade = null;
    shown = PAGE_SIZE;
    render();
  }

  function chooseDecade(decade) {
    clearSearch(true);
    activeAlbumId = "";
    activeDecade = decade;
    shown = PAGE_SIZE;
    render();
    elements.photosHeading.scrollIntoView({ behavior: "smooth" });
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

    const allYears = document.createElement("button");
    allYears.type = "button";
    allYears.className = "decade-button";
    allYears.dataset.decade = "";
    allYears.textContent = "All years";
    allYears.addEventListener("click", () => chooseDecade(null));
    elements.decadeFilter.append(allYears);

    const decades = [...new Set(
      data.photos
        .map((photo) => photo.date_start ? Math.floor(photo.date_start / 10) * 10 : null)
        .filter(Boolean),
    )].sort();
    decades.forEach((decade) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "decade-button";
      button.dataset.decade = String(decade);
      button.textContent = `${decade}s`;
      button.addEventListener("click", () => chooseDecade(decade));
      elements.decadeFilter.append(button);
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
        activeAlbumId = album.id;
        activeDecade = null;
        shown = PAGE_SIZE;
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
        button.addEventListener("click", () => {
          elements.semanticQuery.value = query;
          elements.semanticForm.requestSubmit();
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
    filteredPhotos = data.photos.filter((photo) => {
      if (activeAlbumId && photo.album_id !== activeAlbumId) return false;
      if (activeDecade && (!photo.date_start || Math.floor(photo.date_start / 10) * 10 !== activeDecade)) return false;
      return true;
    });
  }

  function stateText() {
    const album = albums.get(activeAlbumId);
    const within = album ? ` within ${album.title}` : (activeDecade ? ` within the ${activeDecade}s` : "");
    if (searchMode === "text") return `Searching for “${searchQuery}”${within}`;
    if (searchMode === "face") return "Showing photographs of this person";
    if (searchMode === "similar") return "Showing photographs similar to the selected photograph";
    if (album) return `Showing photographs from ${album.title}`;
    if (activeDecade) return `Showing photographs from the ${activeDecade}s`;
    return "";
  }

  function renderHeading() {
    const album = albums.get(activeAlbumId);
    if (searchMode === "text") {
      elements.albumLabel.textContent = "Search results";
      elements.photosHeading.textContent = `“${searchQuery}”`;
      elements.description.textContent = searchDescription;
    } else if (searchMode === "face") {
      elements.albumLabel.textContent = "Search results";
      elements.photosHeading.textContent = "Photographs of this person";
      elements.description.textContent = searchDescription;
    } else if (searchMode === "similar") {
      elements.albumLabel.textContent = "Search results";
      elements.photosHeading.textContent = "Similar photographs";
      elements.description.textContent = searchDescription;
    } else if (album) {
      elements.albumLabel.textContent = "Album";
      elements.photosHeading.textContent = album.title;
      elements.description.textContent = album.description;
    } else if (activeDecade) {
      elements.albumLabel.textContent = "Decade";
      elements.photosHeading.textContent = `${activeDecade}s photographs`;
      elements.description.textContent = "";
    } else {
      elements.albumLabel.textContent = "All albums";
      elements.photosHeading.textContent = "Photographs";
      elements.description.textContent = "";
    }
  }

  function render() {
    applyFilters();
    elements.grid.replaceChildren();
    renderHeading();
    const currentState = stateText();
    elements.resultState.hidden = !currentState;
    elements.resultStateText.textContent = currentState;
    elements.resultsSection.classList.toggle("search-results", searchPhotos !== null);
    elements.count.textContent = plural(filteredPhotos.length, "photograph");
    elements.empty.hidden = filteredPhotos.length !== 0;
    elements.loadMore.hidden = shown >= filteredPhotos.length;
    elements.backToSearch.hidden = !(currentState || shown > PAGE_SIZE);

    document.querySelectorAll(".album-card").forEach((card) => {
      card.classList.toggle("active", card.dataset.albumId === activeAlbumId);
    });
    document.querySelectorAll(".decade-button").forEach((button) => {
      button.classList.toggle(
        "active",
        button.dataset.decade === (activeDecade ? String(activeDecade) : ""),
      );
    });

    filteredPhotos.slice(0, shown).forEach((photo, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "photo-card";
      button.title = photo.filename;
      const image = document.createElement("img");
      image.src = photo.thumbnail;
      image.alt = photo.filename;
      image.loading = "lazy";
      image.decoding = "async";
      if (photo.width && photo.height) button.style.aspectRatio = `${photo.width} / ${photo.height}`;
      button.append(image);
      button.addEventListener("click", () => openViewer(index));
      elements.grid.append(button);
    });
  }

  function openViewer(index) {
    viewerIndex = index;
    const photo = filteredPhotos[index];
    const album = albums.get(photo.album_id);
    elements.viewerImage.src = photo.preview;
    elements.viewerImage.alt = photo.filename;
    elements.viewerTitle.textContent = photo.filename;
    elements.viewerAlbum.textContent = album.title;
    elements.faceLayer.replaceChildren();
    elements.viewerStatus.textContent = "";
    elements.findSimilar.disabled = false;
    elements.findPerson.disabled = false;
    elements.findSimilar.hidden = !imageSearchAvailable;
    elements.findPerson.hidden = !faceSearchAvailable;
    if (photo.original) {
      elements.original.href = photo.original;
      elements.original.hidden = false;
    } else {
      elements.original.hidden = true;
    }
    if (!elements.viewer.open) elements.viewer.showModal();
  }

  function showSearchResults(results, mode, query, description) {
    searchPhotos = results.map((result) => photosById.get(result.id)).filter(Boolean);
    searchMode = mode;
    searchQuery = query || "";
    searchDescription = description;
    shown = PAGE_SIZE;
    render();
    elements.photosHeading.scrollIntoView({ behavior: "smooth" });
  }

  async function enableSemanticSearch() {
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (!response.ok) return;
      const status = await response.json();
      if (!status.available) return;
      imageSearchAvailable = true;
      faceSearchAvailable = Boolean(status.face_search);
      elements.semanticForm.hidden = false;
      elements.semanticStatus.textContent = `Ready to search ${status.photo_count.toLocaleString()} photographs.`;
    } catch (_) {
      // Direct file:// browsing intentionally keeps model-backed controls hidden.
    }
  }

  elements.semanticForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const query = elements.semanticQuery.value.trim();
    if (!query) return;
    elements.semanticSubmit.disabled = true;
    elements.semanticStatus.classList.remove("error");
    elements.semanticStatus.textContent = "Searching photographs…";
    try {
      const response = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          top_k: 100,
          album: activeAlbumId || null,
          date_from: activeDecade,
          date_to: activeDecade ? activeDecade + 9 : null,
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Search failed");
      showSearchResults(payload.results, "text", query, "Ranked by visual similarity.");
      elements.semanticStatus.textContent = `${plural(payload.results.length, "result")} ranked by visual similarity.`;
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
    elements.findPerson.disabled = true;
    elements.viewerStatus.textContent = "Finding similar photographs…";
    try {
      const response = await fetch("/api/similar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ photo_id: photo.id, top_k: 100 }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Similar-photo search failed");
      activeAlbumId = "";
      activeDecade = null;
      elements.viewer.close();
      showSearchResults(payload.results, "similar", "", "Ranked by whole-photograph similarity.");
    } catch (error) {
      elements.viewerStatus.textContent = error.message || "Similar-photo search failed";
      elements.findSimilar.disabled = false;
      elements.findPerson.disabled = false;
    }
  });

  async function searchForFace(faceId) {
    elements.findSimilar.disabled = true;
    elements.findPerson.disabled = true;
    elements.faceLayer.replaceChildren();
    elements.viewerStatus.textContent = "Finding this person…";
    try {
      const response = await fetch("/api/face-search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ face_id: faceId, top_k: 100 }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Face search failed");
      activeAlbumId = "";
      activeDecade = null;
      elements.viewer.close();
      showSearchResults(payload.results, "face", "", "Ranked by face similarity.");
    } catch (error) {
      elements.viewerStatus.textContent = error.message || "Face search failed";
      elements.findSimilar.disabled = false;
      elements.findPerson.disabled = false;
    }
  }

  elements.findPerson.addEventListener("click", async () => {
    const photo = filteredPhotos[viewerIndex];
    if (!photo) return;
    elements.findSimilar.disabled = true;
    elements.findPerson.disabled = true;
    elements.faceLayer.replaceChildren();
    elements.viewerStatus.textContent = "Looking for faces…";
    try {
      const response = await fetch("/api/faces", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ photo_id: photo.id }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not load faces");
      if (!payload.faces.length) {
        elements.viewerStatus.textContent = "No clear face was found in this photograph.";
        elements.findSimilar.disabled = false;
        elements.findPerson.disabled = false;
        return;
      }
      if (payload.faces.length === 1) {
        await searchForFace(payload.faces[0].id);
        return;
      }
      payload.faces.forEach((face, index) => {
        const [x, y, width, height] = face.bbox;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "face-choice";
        button.setAttribute("aria-label", `Find person ${index + 1}`);
        button.style.left = `${x * 100}%`;
        button.style.top = `${y * 100}%`;
        button.style.width = `${width * 100}%`;
        button.style.height = `${height * 100}%`;
        button.addEventListener("click", () => searchForFace(face.id));
        elements.faceLayer.append(button);
      });
      elements.viewerStatus.textContent = "Choose a face in the photograph.";
      elements.findSimilar.disabled = false;
      elements.findPerson.disabled = false;
    } catch (error) {
      elements.viewerStatus.textContent = error.message || "Could not load faces";
      elements.findSimilar.disabled = false;
      elements.findPerson.disabled = false;
    }
  });

  elements.clear.addEventListener("click", clearEverything);
  elements.clearResultState.addEventListener("click", clearEverything);
  elements.loadMore.addEventListener("click", () => { shown += PAGE_SIZE; render(); });
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

(() => {
  "use strict";

  const data = window.FAMILY_PHOTO_DATA;
  const PAGE_SIZE = 80;
  const albums = new Map(data.albums.map((album) => [album.id, album]));
  const elements = {
    summary: document.querySelector("#collection-summary"),
    hero: document.querySelector("#hero-strip"),
    semanticForm: document.querySelector("#semantic-search-form"),
    semanticQuery: document.querySelector("#semantic-query"),
    semanticSubmit: document.querySelector("#semantic-submit"),
    semanticStatus: document.querySelector("#semantic-status"),
    albumFilter: document.querySelector("#album-filter"),
    decadeFilter: document.querySelector("#decade-filter"),
    clear: document.querySelector("#clear-filters"),
    albumList: document.querySelector("#album-list"),
    albumLabel: document.querySelector("#active-album-label"),
    description: document.querySelector("#active-description"),
    count: document.querySelector("#result-count"),
    grid: document.querySelector("#photo-grid"),
    empty: document.querySelector("#empty-state"),
    loadMore: document.querySelector("#load-more"),
    viewer: document.querySelector("#viewer"),
    viewerImage: document.querySelector("#viewer-image"),
    viewerTitle: document.querySelector("#viewer-title"),
    viewerAlbum: document.querySelector("#viewer-album"),
    original: document.querySelector("#open-original"),
  };

  let filteredPhotos = data.photos;
  let shown = PAGE_SIZE;
  let viewerIndex = -1;
  let semanticPhotos = null;
  let semanticQuery = "";
  const photosById = new Map(data.photos.map((photo) => [photo.id, photo]));

  function plural(number, word) {
    return `${number.toLocaleString()} ${word}${number === 1 ? "" : "s"}`;
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
    data.albums.forEach((album) => {
      const option = document.createElement("option");
      option.value = album.id;
      option.textContent = album.title;
      elements.albumFilter.append(option);

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
        elements.albumFilter.value = album.id;
        elements.decadeFilter.value = "";
        shown = PAGE_SIZE;
        render();
        document.querySelector("#photos-heading").scrollIntoView({ behavior: "smooth" });
      });
      elements.albumList.append(card);
    });

    const decades = [...new Set(data.photos.map((photo) => photo.date_start ? Math.floor(photo.date_start / 10) * 10 : null).filter(Boolean))].sort();
    decades.forEach((decade) => {
      const option = document.createElement("option");
      option.value = String(decade);
      option.textContent = `${decade}s`;
      elements.decadeFilter.append(option);
    });
  }

  function applyFilters() {
    if (semanticPhotos) {
      filteredPhotos = semanticPhotos;
      return;
    }
    const albumId = elements.albumFilter.value;
    const decade = Number(elements.decadeFilter.value || 0);
    filteredPhotos = data.photos.filter((photo) => {
      if (albumId && photo.album_id !== albumId) return false;
      if (decade && (!photo.date_start || Math.floor(photo.date_start / 10) * 10 !== decade)) return false;
      return true;
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
    if (photo.original) {
      elements.original.href = photo.original;
      elements.original.hidden = false;
    } else {
      elements.original.hidden = true;
    }
    if (!elements.viewer.open) elements.viewer.showModal();
  }

  function render() {
    applyFilters();
    elements.grid.replaceChildren();
    const selectedAlbum = albums.get(elements.albumFilter.value);
    const selectedDecade = elements.decadeFilter.value;
    elements.albumLabel.textContent = semanticQuery ? `Search: ${semanticQuery}` : (selectedAlbum ? selectedAlbum.title : (selectedDecade ? `${selectedDecade}s` : "All albums"));
    elements.description.textContent = semanticQuery ? "Ranked by visual similarity." : (selectedAlbum ? selectedAlbum.description : "");
    elements.count.textContent = plural(filteredPhotos.length, "photograph");
    elements.empty.hidden = filteredPhotos.length !== 0;
    elements.loadMore.hidden = shown >= filteredPhotos.length;

    document.querySelectorAll(".album-card").forEach((card) => {
      card.classList.toggle("active", card.dataset.albumId === elements.albumFilter.value);
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

  elements.albumFilter.addEventListener("change", () => {
    elements.decadeFilter.value = "";
    clearSemanticResults();
    shown = PAGE_SIZE;
    render();
  });
  elements.decadeFilter.addEventListener("change", () => {
    elements.albumFilter.value = "";
    clearSemanticResults();
    shown = PAGE_SIZE;
    render();
  });
  elements.clear.addEventListener("click", () => {
    clearSemanticResults(true);
    elements.albumFilter.value = "";
    elements.decadeFilter.value = "";
    shown = PAGE_SIZE;
    render();
  });
  elements.loadMore.addEventListener("click", () => { shown += PAGE_SIZE; render(); });

  function clearSemanticResults(clearInput = false) {
    semanticPhotos = null;
    semanticQuery = "";
    if (clearInput && elements.semanticQuery) elements.semanticQuery.value = "";
    if (elements.semanticStatus) {
      elements.semanticStatus.textContent = "Search looks at the photographs themselves.";
      elements.semanticStatus.classList.remove("error");
    }
  }

  async function enableSemanticSearch() {
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (!response.ok) return;
      const status = await response.json();
      if (!status.available) return;
      elements.semanticForm.hidden = false;
      elements.semanticStatus.textContent = `Ready to search ${status.photo_count.toLocaleString()} photographs.`;
    } catch (_) {
      // Direct file:// browsing intentionally keeps semantic controls hidden.
    }
  }

  elements.semanticForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const query = elements.semanticQuery.value.trim();
    if (!query) return;
    elements.semanticSubmit.disabled = true;
    elements.semanticStatus.classList.remove("error");
    elements.semanticStatus.textContent = "Searching photographs…";
    const decade = Number(elements.decadeFilter.value || 0);
    try {
      const response = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          top_k: 100,
          album: elements.albumFilter.value || null,
          date_from: decade || null,
          date_to: decade ? decade + 9 : null,
        }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Search failed");
      semanticPhotos = payload.results.map((result) => photosById.get(result.id)).filter(Boolean);
      semanticQuery = query;
      shown = PAGE_SIZE;
      elements.semanticStatus.textContent = `${plural(semanticPhotos.length, "result")} ranked by visual similarity.`;
      render();
      document.querySelector("#photos-heading").scrollIntoView({ behavior: "smooth" });
    } catch (error) {
      elements.semanticStatus.textContent = error.message || "Search failed";
      elements.semanticStatus.classList.add("error");
    } finally {
      elements.semanticSubmit.disabled = false;
    }
  });
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
  render();
  enableSemanticSearch();
})();

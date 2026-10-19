// Similar-photo and face search over precomputed vectors, for use without the
// local server. Ranking mirrors SemanticSearchEngine.similar and
// FaceSearchIndex.search in src/search.py.
(() => {
  "use strict";

  function decodeVectors(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return new Float32Array(bytes.buffer);
  }

  function scoreAll(vectors, dimension, row) {
    const count = vectors.length / dimension;
    const scores = new Float32Array(count);
    const offset = row * dimension;
    for (let candidate = 0; candidate < count; candidate += 1) {
      const start = candidate * dimension;
      let total = 0;
      for (let position = 0; position < dimension; position += 1) {
        total += vectors[start + position] * vectors[offset + position];
      }
      scores[candidate] = total;
    }
    return scores;
  }

  function rankedRows(scores) {
    const rows = Array.from(scores.keys());
    rows.sort((left, right) => scores[right] - scores[left]);
    return rows;
  }

  function createSimilarIndex(payload) {
    const vectors = decodeVectors(payload.vectors);
    const rows = new Map(payload.ids.map((id, row) => [id, row]));
    return {
      similar(photoId, topK) {
        const row = rows.get(photoId);
        if (row === undefined) throw new Error("Photo not found");
        const scores = scoreAll(vectors, payload.dimension, row);
        return rankedRows(scores)
          .filter((candidate) => candidate !== row)
          .slice(0, topK)
          .map((candidate) => ({ id: payload.ids[candidate], score: scores[candidate] }));
      },
    };
  }

  function createFaceIndex(payload) {
    const vectors = decodeVectors(payload.vectors);
    const rows = new Map(payload.ids.map((id, row) => [id, row]));
    const facesByPhoto = new Map();
    payload.ids.forEach((id, row) => {
      const photoId = payload.photo_ids[row];
      if (!facesByPhoto.has(photoId)) facesByPhoto.set(photoId, []);
      facesByPhoto.get(photoId).push({ id, bbox: payload.bboxes[row] });
    });
    return {
      facesForPhoto(photoId) {
        return facesByPhoto.get(photoId) || [];
      },
      search(faceId, topK, knownPhotoIds) {
        const row = rows.get(faceId);
        if (row === undefined) throw new Error("Face not found");
        const scores = scoreAll(vectors, payload.dimension, row);
        const seenPhotos = new Set([payload.photo_ids[row]]);
        const results = [];
        for (const candidate of rankedRows(scores)) {
          const photoId = payload.photo_ids[candidate];
          if (seenPhotos.has(photoId) || !knownPhotoIds.has(photoId)) continue;
          seenPhotos.add(photoId);
          results.push({
            id: photoId,
            score: scores[candidate],
            matched_face: payload.bboxes[candidate],
          });
          if (results.length >= topK) break;
        }
        return results;
      },
    };
  }

  window.MemoryLaneBrowserSearch = { decodeVectors, createSimilarIndex, createFaceIndex };
})();

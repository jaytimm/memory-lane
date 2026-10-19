// CLIP text encoder for typed search without the local server. It mirrors
// open_clip's SimpleTokenizer and CLIP.encode_text, using the int8 weights
// exported by src/text_model.py.
(() => {
  "use strict";

  function decodeBytes(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return bytes;
  }

  function loadTensor(entry) {
    if (entry.float32) return { shape: entry.shape, values: new Float32Array(decodeBytes(entry.float32).buffer) };
    return {
      shape: entry.shape,
      quantized: new Int8Array(decodeBytes(entry.int8).buffer),
      scales: new Float32Array(decodeBytes(entry.scales).buffer),
    };
  }

  // open_clip.tokenizer.bytes_to_unicode
  function bytesToUnicode() {
    const bytes = [];
    for (let code = 33; code <= 126; code += 1) bytes.push(code);
    for (let code = 161; code <= 172; code += 1) bytes.push(code);
    for (let code = 174; code <= 255; code += 1) bytes.push(code);
    const characters = bytes.slice();
    let extra = 0;
    for (let code = 0; code < 256; code += 1) {
      if (!bytes.includes(code)) {
        bytes.push(code);
        characters.push(256 + extra);
        extra += 1;
      }
    }
    const table = new Array(256);
    bytes.forEach((code, index) => { table[code] = String.fromCharCode(characters[index]); });
    return table;
  }

  const HTML_ENTITIES = { amp: "&", lt: "<", gt: ">", quot: "\"", apos: "'", nbsp: " " };

  function unescapeHtml(text) {
    return text.replace(/&(#\d+|#x[0-9a-f]+|amp|lt|gt|quot|apos|nbsp);/gi, (match, name) => {
      const lower = name.toLowerCase();
      if (lower.startsWith("#x")) return String.fromCodePoint(parseInt(lower.slice(2), 16));
      if (lower.startsWith("#")) return String.fromCodePoint(parseInt(lower.slice(1), 10));
      return HTML_ENTITIES[lower];
    });
  }

  // open_clip's "lower" cleaning. ftfy.fix_text is approximated by the parts
  // that matter for typed text: NFC, straight quotes, ligatures, full width.
  function cleanText(text) {
    let cleaned = text.normalize("NFC")
      .replace(/[ʼ‘-‛]/g, "'")
      .replace(/[“-‟]/g, "\"")
      .replace(/[ﬀ-ﬆ！-～]/g, (character) => character.normalize("NFKC"));
    cleaned = unescapeHtml(unescapeHtml(cleaned)).trim();
    return cleaned.split(/\s+/).filter(Boolean).join(" ").toLowerCase();
  }

  function createTokenizer(model) {
    const encoder = new Map(model.vocab.map((token, id) => [token, id]));
    const ranks = new Map(model.merges.map((merge, rank) => [merge, rank]));
    const byteEncoder = bytesToUnicode();
    const special = new Map([["<start_of_text>", model.sot], ["<end_of_text>", model.eot]]);
    const pattern = /<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/giu;
    const utf8 = new TextEncoder();
    const cache = new Map();

    function bpe(token) {
      if (cache.has(token)) return cache.get(token);
      let word = Array.from(token);
      word[word.length - 1] += "</w>";
      while (word.length > 1) {
        let best = null;
        let bestRank = Infinity;
        for (let index = 0; index < word.length - 1; index += 1) {
          const rank = ranks.get(`${word[index]} ${word[index + 1]}`);
          if (rank !== undefined && rank < bestRank) {
            bestRank = rank;
            best = index;
          }
        }
        if (best === null) break;
        const first = word[best];
        const second = word[best + 1];
        const merged = [];
        for (let index = 0; index < word.length;) {
          if (index < word.length - 1 && word[index] === first && word[index + 1] === second) {
            merged.push(first + second);
            index += 2;
          } else {
            merged.push(word[index]);
            index += 1;
          }
        }
        word = merged;
      }
      cache.set(token, word);
      return word;
    }

    return function tokenize(text) {
      const ids = [model.sot];
      for (const [piece] of cleanText(text).matchAll(pattern)) {
        if (special.has(piece)) {
          ids.push(special.get(piece));
          continue;
        }
        const encoded = Array.from(utf8.encode(piece), (byte) => byteEncoder[byte]).join("");
        for (const symbol of bpe(encoded)) ids.push(encoder.get(symbol));
      }
      ids.push(model.eot);
      if (ids.length > model.context_length) {
        ids.length = model.context_length;
        ids[ids.length - 1] = model.eot;
      }
      return ids;
    };
  }

  // y[t, row] = W[row, :] · x[t, :] + bias[row] for every token t. Each row
  // is dequantized once, then shared by all tokens.
  function linear(input, tokens, weight, bias, groupSize) {
    const [rows, columns] = weight.shape;
    const output = new Float32Array(tokens * rows);
    const groups = columns / groupSize;
    const { quantized, scales } = weight;
    const row32 = new Float32Array(columns);
    for (let row = 0; row < rows; row += 1) {
      const rowStart = row * columns;
      for (let group = 0; group < groups; group += 1) {
        const scale = scales[row * groups + group];
        const end = (group + 1) * groupSize;
        for (let index = group * groupSize; index < end; index += 1) {
          row32[index] = quantized[rowStart + index] * scale;
        }
      }
      for (let token = 0; token < tokens; token += 1) {
        const offset = token * columns;
        let a = 0;
        let b = 0;
        let c = 0;
        let d = 0;
        for (let index = 0; index < columns; index += 4) {
          a += row32[index] * input[offset + index];
          b += row32[index + 1] * input[offset + index + 1];
          c += row32[index + 2] * input[offset + index + 2];
          d += row32[index + 3] * input[offset + index + 3];
        }
        output[token * rows + row] = a + b + c + d + bias.values[row];
      }
    }
    return output;
  }

  function layerNorm(input, tokens, width, weight, bias) {
    const output = new Float32Array(input.length);
    for (let token = 0; token < tokens; token += 1) {
      const start = token * width;
      let mean = 0;
      for (let index = 0; index < width; index += 1) mean += input[start + index];
      mean /= width;
      let variance = 0;
      for (let index = 0; index < width; index += 1) variance += (input[start + index] - mean) ** 2;
      const inverse = 1 / Math.sqrt(variance / width + 1e-5);
      for (let index = 0; index < width; index += 1) {
        output[start + index] = (input[start + index] - mean) * inverse * weight.values[index] + bias.values[index];
      }
    }
    return output;
  }

  // Abramowitz and Stegun 7.1.26; absolute error below 1.5e-7.
  function erf(value) {
    const sign = value < 0 ? -1 : 1;
    const x = Math.abs(value);
    const t = 1 / (1 + 0.3275911 * x);
    const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
    return sign * (1 - poly * Math.exp(-x * x));
  }

  // Attention outputs for query positions firstQuery..tokens-1.
  function causalAttention(qkv, tokens, width, heads, firstQuery) {
    const headSize = width / heads;
    const scale = 1 / Math.sqrt(headSize);
    const output = new Float32Array((tokens - firstQuery) * width);
    const weights = new Float64Array(tokens);
    for (let head = 0; head < heads; head += 1) {
      const headOffset = head * headSize;
      for (let query = firstQuery; query < tokens; query += 1) {
        const queryStart = query * 3 * width + headOffset;
        let largest = -Infinity;
        for (let key = 0; key <= query; key += 1) {
          const keyStart = key * 3 * width + width + headOffset;
          let score = 0;
          for (let index = 0; index < headSize; index += 1) score += qkv[queryStart + index] * qkv[keyStart + index];
          weights[key] = score * scale;
          if (weights[key] > largest) largest = weights[key];
        }
        let total = 0;
        for (let key = 0; key <= query; key += 1) {
          weights[key] = Math.exp(weights[key] - largest);
          total += weights[key];
        }
        const outputStart = (query - firstQuery) * width + headOffset;
        for (let key = 0; key <= query; key += 1) {
          const valueStart = key * 3 * width + 2 * width + headOffset;
          const weight = weights[key] / total;
          for (let index = 0; index < headSize; index += 1) {
            output[outputStart + index] += weight * qkv[valueStart + index];
          }
        }
      }
    }
    return output;
  }

  function createTextEncoder(model) {
    const tokenize = createTokenizer(model);
    const tensors = {};
    for (const [name, entry] of Object.entries(model.tensors)) tensors[name] = loadTensor(entry);
    const { width, heads, layers, group_size: groupSize } = model;
    // Keys and values for a position depend only on the tokens up to it, so
    // the shared "a photo of" prefix (and any prefix shared with the previous
    // query) is reused instead of recomputed.
    let cached = { ids: [], qkv: [] };

    const encoder = {
      encode(query) {
        // CLIP pools the first EOT token, which has the largest id. Attention
        // is causal, so later positions cannot change it and are skipped.
        const allIds = tokenize(model.prompt.replace("{}", query.trim()));
        const ids = allIds.slice(0, allIds.indexOf(model.eot) + 1);
        const tokens = ids.length;
        let reused = 0;
        while (reused < tokens - 1 && ids[reused] === cached.ids[reused]) reused += 1;
        const fresh = tokens - reused;

        const embedding = tensors["token_embedding.weight"];
        const positions = tensors.positional_embedding.values;
        const groups = width / groupSize;
        let x = new Float32Array(fresh * width);
        for (let token = 0; token < fresh; token += 1) {
          const id = ids[reused + token];
          const position = reused + token;
          for (let index = 0; index < width; index += 1) {
            const scale = embedding.scales[id * groups + Math.floor(index / groupSize)];
            x[token * width + index] = embedding.quantized[id * width + index] * scale + positions[position * width + index];
          }
        }

        const qkvByLayer = [];
        for (let layer = 0; layer < layers; layer += 1) {
          const prefix = `transformer.resblocks.${layer}.`;
          const tensor = (name) => tensors[prefix + name];
          let hidden = layerNorm(x, fresh, width, tensor("ln_1.weight"), tensor("ln_1.bias"));
          const qkv = new Float32Array(tokens * 3 * width);
          if (reused) qkv.set(cached.qkv[layer].subarray(0, reused * 3 * width));
          qkv.set(linear(hidden, fresh, tensor("attn.in_proj_weight"), tensor("attn.in_proj_bias"), groupSize), reused * 3 * width);
          qkvByLayer.push(qkv);
          const attended = causalAttention(qkv, tokens, width, heads, reused);
          const projected = linear(attended, fresh, tensor("attn.out_proj.weight"), tensor("attn.out_proj.bias"), groupSize);
          for (let index = 0; index < x.length; index += 1) x[index] += projected[index];

          hidden = layerNorm(x, fresh, width, tensor("ln_2.weight"), tensor("ln_2.bias"));
          hidden = linear(hidden, fresh, tensor("mlp.c_fc.weight"), tensor("mlp.c_fc.bias"), groupSize);
          for (let index = 0; index < hidden.length; index += 1) {
            hidden[index] = 0.5 * hidden[index] * (1 + erf(hidden[index] / Math.SQRT2));
          }
          const mlp = linear(hidden, fresh, tensor("mlp.c_proj.weight"), tensor("mlp.c_proj.bias"), groupSize);
          for (let index = 0; index < x.length; index += 1) x[index] += mlp[index];
        }
        cached = { ids, qkv: qkvByLayer };

        const last = x.subarray((fresh - 1) * width, fresh * width);
        const pooled = layerNorm(last, 1, width, tensors["ln_final.weight"], tensors["ln_final.bias"]);
        const projection = tensors.text_projection;
        const outputSize = projection.shape[1];
        const vector = new Float32Array(outputSize);
        for (let input = 0; input < width; input += 1) {
          for (let output = 0; output < outputSize; output += 1) {
            vector[output] += pooled[input] * projection.values[input * outputSize + output];
          }
        }
        let norm = 0;
        for (const value of vector) norm += value * value;
        norm = Math.sqrt(norm);
        for (let index = 0; index < outputSize; index += 1) vector[index] /= norm;
        return vector;
      },
      tokenize,
    };
    encoder.encode("");
    return encoder;
  }

  window.MemoryLaneTextEncoder = { createTextEncoder, cleanText };
})();

// GLNet: tiny WebGL2 inference engine for U²-Net (silueta) — no WebAssembly, no workers.
// Tensors are RGBA float textures; 4 channels per texel, channel groups laid out as tiles.
const GLNet = (() => {
  const VS = `#version 300 es
in vec2 p; void main(){ gl_Position = vec4(p, 0.0, 1.0); }`;
  const HEAD = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D;
out vec4 o;
// tensor descriptor: x=W y=H z=C4 w=tiles-per-row
ivec2 tileBase(int t, ivec4 d){ return ivec2((t % d.w) * d.x, (t / d.w) * d.y); }
bool locate(ivec4 d, out int t, out ivec2 p){
  ivec2 fc = ivec2(gl_FragCoord.xy); int tx = fc.x / d.x, ty = fc.y / d.y;
  t = ty * d.w + tx; p = fc - ivec2(tx * d.x, ty * d.y); return t < d.z;
}
`;
  const FS = {
    conv: HEAD + `
uniform sampler2D X, Wt, Bt; uniform ivec4 xd, yd; uniform int K, dil, pad, relu, woff, boff, wtw; uniform float scale, zp;
vec4 fw(int i){ vec4 q = floor(texelFetch(Wt, ivec2(i % wtw, i / wtw), 0) * 255.0 + 0.5); return (q - zp) * scale; }
void main(){
  int t; ivec2 p; if (!locate(yd, t, p)) { o = vec4(0.0); return; }
  vec4 acc = texelFetch(Bt, ivec2((boff + t) % 1024, (boff + t) / 1024), 0);
  int KK = K * K; int wbase = woff + t * xd.z * KK * 4;
  for (int ci = 0; ci < xd.z; ci++) {
    ivec2 tb = tileBase(ci, xd);
    for (int ky = 0; ky < K; ky++) {
      int iy = p.y - pad + ky * dil;
      if (iy < 0 || iy >= xd.y) continue;
      for (int kx = 0; kx < K; kx++) {
        int ix = p.x - pad + kx * dil;
        if (ix < 0 || ix >= xd.x) continue;
        vec4 v = texelFetch(X, tb + ivec2(ix, iy), 0);
        int wi = wbase + ((ci * KK) + ky * K + kx) * 4;
        acc += v * mat4(fw(wi), fw(wi + 1), fw(wi + 2), fw(wi + 3));
      }
    }
  }
  o = relu == 1 ? max(acc, vec4(0.0)) : acc;
}`,
    pool: HEAD + `
uniform sampler2D X; uniform ivec4 xd, yd;
void main(){
  int t; ivec2 p; if (!locate(yd, t, p)) { o = vec4(0.0); return; }
  ivec2 tb = tileBase(t, xd); vec4 m = vec4(-3.0e38);
  for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
    ivec2 q = p * 2 + ivec2(dx, dy);
    if (q.x < xd.x && q.y < xd.y) m = max(m, texelFetch(X, tb + q, 0));
  }
  o = m;
}`,
    resize: HEAD + `
uniform sampler2D X; uniform ivec4 xd, yd;
void main(){
  int t; ivec2 p; if (!locate(yd, t, p)) { o = vec4(0.0); return; }
  vec2 s = vec2(xd.xy) / vec2(yd.xy);
  vec2 c = (vec2(p) + 0.5) * s - 0.5;
  c = clamp(c, vec2(0.0), vec2(xd.xy - 1));
  ivec2 i0 = ivec2(floor(c)); ivec2 i1 = min(i0 + 1, xd.xy - 1); vec2 f = c - vec2(i0);
  ivec2 tb = tileBase(t, xd);
  vec4 a = mix(texelFetch(X, tb + ivec2(i0.x, i0.y), 0), texelFetch(X, tb + ivec2(i1.x, i0.y), 0), f.x);
  vec4 b = mix(texelFetch(X, tb + ivec2(i0.x, i1.y), 0), texelFetch(X, tb + ivec2(i1.x, i1.y), 0), f.x);
  o = mix(a, b, f.y);
}`,
    concat: HEAD + `
uniform sampler2D A, B; uniform ivec4 ad, bd, yd; uniform int ca;
float lane(sampler2D S, ivec4 d, int c, ivec2 p){ vec4 v = texelFetch(S, tileBase(c / 4, d) + p, 0); int l = c % 4; return l == 0 ? v.x : l == 1 ? v.y : l == 2 ? v.z : v.w; }
void main(){
  int t; ivec2 p; if (!locate(yd, t, p)) { o = vec4(0.0); return; }
  if (ca % 4 == 0) {
    o = t < ca / 4 ? texelFetch(A, tileBase(t, ad) + p, 0) : texelFetch(B, tileBase(t - ca / 4, bd) + p, 0);
    return;
  }
  vec4 r;
  for (int j = 0; j < 4; j++) { int c = t * 4 + j; r[j] = c < ca ? lane(A, ad, c, p) : lane(B, bd, c - ca, p); }
  o = r;
}`,
    add: HEAD + `
uniform sampler2D A, B; uniform ivec4 ad, bd, yd;
void main(){ int t; ivec2 p; if (!locate(yd, t, p)) { o = vec4(0.0); return; }
  o = texelFetch(A, tileBase(t, ad) + p, 0) + texelFetch(B, tileBase(t, bd) + p, 0); }`,
    sigmoid: HEAD + `
uniform sampler2D X; uniform ivec4 xd, yd;
void main(){ int t; ivec2 p; if (!locate(yd, t, p)) { o = vec4(0.0); return; }
  o = 1.0 / (1.0 + exp(-texelFetch(X, tileBase(t, xd) + p, 0))); }`,
    out8: `#version 300 es
precision highp float; uniform sampler2D X; out vec4 o;
void main(){ float v = texelFetch(X, ivec2(gl_FragCoord.xy), 0).x; o = vec4(clamp(v, 0.0, 1.0), 0.0, 0.0, 1.0); }`,
  };

  class Net {
    constructor(meta, weights, bias) { this.meta = meta; this.weights = weights; this.bias = bias; this.ready = false; }

    init() {
      const c = document.createElement("canvas"); c.width = c.height = 1;
      const gl = c.getContext("webgl2", { antialias: false, depth: false, stencil: false, premultipliedAlpha: false, powerPreference: "high-performance" });
      if (!gl) throw new Error("เบราว์เซอร์นี้ไม่รองรับ WebGL2");
      this.gl = gl;
      if (gl.getExtension("EXT_color_buffer_float")) { this.ifmt = gl.RGBA32F; this.type = gl.FLOAT; }
      else if (gl.getExtension("EXT_color_buffer_half_float")) { this.ifmt = gl.RGBA16F; this.type = gl.HALF_FLOAT; }
      else throw new Error("การ์ดจอไม่รองรับการคำนวณแบบทศนิยม");
      this.maxTex = Math.min(gl.getParameter(gl.MAX_TEXTURE_SIZE), 8192);
      const [ww, wh] = this.meta.wtex;
      if (ww > this.maxTex || wh > this.maxTex) throw new Error("การ์ดจอรองรับขนาดเท็กซ์เจอร์ไม่พอ");
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      this.wtex = this.tex(ww, wh, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, this.weights);
      const [bw, bh] = this.meta.btex;
      this.btex = this.tex(bw, bh, gl.RGBA32F, gl.RGBA, gl.FLOAT, this.bias);
      this.weights = null;
      const vb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vb);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      this.progs = {};
      for (const k in FS) this.progs[k] = this.program(FS[k]);
      this.fb = gl.createFramebuffer();
      this.pool = new Map();
      this.ready = true;
    }

    tex(w, h, ifmt, fmt, type, data) {
      const gl = this.gl, t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, data || null);
      return t;
    }

    program(fs) {
      const gl = this.gl;
      const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error("shader: " + gl.getShaderInfoLog(s)); return s; };
      const p = gl.createProgram();
      gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
      gl.bindAttribLocation(p, 0, "p"); gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error("link: " + gl.getProgramInfoLog(p));
      const u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
      for (let i = 0; i < n; i++) { const a = gl.getActiveUniform(p, i); u[a.name] = gl.getUniformLocation(p, a.name); }
      return { p, u };
    }

    // tensor layout for C channels at H×W
    layout(C, H, W) {
      const C4 = Math.ceil(C / 4);
      const tpr = Math.max(1, Math.min(C4, Math.floor(this.maxTex / W)));
      return { C, H, W, C4, tpr, tw: tpr * W, th: Math.ceil(C4 / tpr) * H };
    }
    alloc(L) {
      const key = L.tw + "x" + L.th, list = this.pool.get(key);
      const tex = (list && list.pop()) || this.tex(L.tw, L.th, this.ifmt, this.gl.RGBA, this.type, null);
      return { ...L, tex };
    }
    free(T) { const key = T.tw + "x" + T.th; if (!this.pool.has(key)) this.pool.set(key, []); this.pool.get(key).push(T.tex); }

    desc(T) { return [T.W, T.H, T.C4, T.tpr]; }

    async run(input, onProgress) {       // input: Float32Array CHW 3×320×320
      if (!this.ready) this.init();
      const gl = this.gl, M = this.meta, S = M.shapes;
      const [C0, H0, W0] = S[0];
      // input tensor (RGBA32F, 3 channels + zero lane)
      const inL = this.layout(C0, H0, W0);
      const data = new Float32Array(inL.tw * inL.th * 4), plane = H0 * W0;
      for (let p = 0; p < plane; p++) for (let c = 0; c < C0; c++) data[p * 4 + c] = input[c * plane + p];
      const T = { 0: { ...inL, tex: this.tex(inL.tw, inL.th, gl.RGBA32F, gl.RGBA, gl.FLOAT, data), own: true } };

      // last use of each tensor, so textures can be recycled
      const last = {};
      M.ops.forEach((op, i) => { for (const k of ["i", "a", "b"]) if (op[k] !== undefined) last[op[k]] = i; (op.ins || []).forEach(t => last[t] = i); });
      last[M.out] = Infinity;

      // cost model for progress + splitting long draws
      const cost = op => { const s = S[op.o]; const L = this.layout(s[0], s[1], s[2]);
        return op.op === "conv" ? L.tw * L.th * Math.ceil(op.cin / 4) * op.k * op.k * 5 : L.tw * L.th * 4; };
      const total = M.ops.reduce((a, op) => a + cost(op), 0);
      let doneCost = 0, sinceWait = 0;
      const BUDGET = 6e7;

      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fb);
      gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST);

      for (let i = 0; i < M.ops.length; i++) {
        const op = M.ops[i], s = S[op.o];
        let out;
        if (op.op === "concat" && op.ins.length > 2) {          // chain pairwise
          let acc = T[op.ins[0]], accC = S[op.ins[0]][0], tmp = [];
          for (let j = 1; j < op.ins.length; j++) {
            const bT = T[op.ins[j]], C = accC + S[op.ins[j]][0];
            const o2 = this.alloc(this.layout(C, s[1], s[2]));
            this.draw("concat", o2, { A: acc, B: bT }, { ca: accC });
            if (j > 1) tmp.push(acc);
            acc = o2; accC = C;
          }
          tmp.forEach(t => this.free(t)); out = acc;
        } else {
          out = this.alloc(this.layout(s[0], s[1], s[2]));
          if (op.op === "conv") {
            const c = cost(op), rowsPerDraw = Math.max(1, Math.floor(out.th * BUDGET / c));
            for (let y = 0; y < out.th; y += rowsPerDraw) {
              this.draw("conv", out, { X: T[op.i] }, { K: op.k, dil: op.dil, pad: op.pad, relu: op.relu, woff: op.woff, boff: op.boff, scale: op.scale, zp: op.zp }, [y, Math.min(rowsPerDraw, out.th - y)]);
              sinceWait += c * Math.min(rowsPerDraw, out.th - y) / out.th;
              if (sinceWait > BUDGET * 2) { await this.wait(); sinceWait = 0; onProgress && onProgress(Math.min(0.99, (doneCost + c * (y + rowsPerDraw) / out.th) / total)); }
            }
          }
          else if (op.op === "pool") this.draw("pool", out, { X: T[op.i] });
          else if (op.op === "resize") this.draw("resize", out, { X: T[op.i] });
          else if (op.op === "concat") this.draw("concat", out, { A: T[op.ins[0]], B: T[op.ins[1]] }, { ca: S[op.ins[0]][0] });
          else if (op.op === "add") this.draw("add", out, { A: T[op.a], B: T[op.b] });
          else if (op.op === "sigmoid") this.draw("sigmoid", out, { X: T[op.i] });
        }
        T[op.o] = out;
        doneCost += cost(op);
        for (const k of ["i", "a", "b"]) if (op[k] !== undefined && last[op[k]] === i && T[op[k]]) { if (T[op[k]].own) gl.deleteTexture(T[op[k]].tex); else this.free(T[op[k]]); delete T[op[k]]; }
        (op.ins || []).forEach(t => { if (last[t] === i && T[t]) { this.free(T[t]); delete T[t]; } });
      }

      // final: render channel 0 into RGBA8 and read back
      const R = T[M.out], [_, H, W] = S[M.out];
      const t8 = this.tex(W, H, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t8, 0);
      gl.viewport(0, 0, W, H);
      const P = this.progs.out8; gl.useProgram(P.p);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, R.tex); gl.uniform1i(P.u.X, 0);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      await this.wait();
      const px = new Uint8Array(W * H * 4); gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px);
      gl.deleteTexture(t8); this.free(R);
      const outF = new Float32Array(W * H);
      for (let p = 0; p < W * H; p++) outF[p] = px[p * 4] / 255;
      onProgress && onProgress(1);
      return outF;       // row 0 = top (texture y=0 is written by fragment y=0 → first row of the tensor)
    }

    draw(name, out, texs, ints = {}, rows) {
      const gl = this.gl, P = this.progs[name];
      gl.useProgram(P.p);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, out.tex, 0);
      gl.viewport(0, 0, out.tw, out.th);
      if (rows) { gl.enable(gl.SCISSOR_TEST); gl.scissor(0, rows[0], out.tw, rows[1]); } else gl.disable(gl.SCISSOR_TEST);
      let unit = 0;
      const bindTex = (uname, tex) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, tex); gl.uniform1i(P.u[uname], unit); unit++; };
      const dname = { X: "xd", A: "ad", B: "bd" };
      for (const k in texs) { bindTex(k, texs[k].tex); if (P.u[dname[k]]) gl.uniform4i(P.u[dname[k]], ...this.desc(texs[k])); }
      if (P.u.yd) gl.uniform4i(P.u.yd, ...this.desc(out));
      if (name === "conv") { bindTex("Wt", this.wtex); bindTex("Bt", this.btex); gl.uniform1i(P.u.wtw, this.meta.wtex[0]); }
      for (const k in ints) if (P.u[k]) (k === "scale" || k === "zp") ? gl.uniform1f(P.u[k], ints[k]) : gl.uniform1i(P.u[k], ints[k]);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    async wait() {
      const gl = this.gl, s = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
      for (;;) {
        const r = gl.clientWaitSync(s, 0, 0);
        if (r === gl.ALREADY_SIGNALED || r === gl.CONDITION_SATISFIED || r === gl.WAIT_FAILED) break;
        await new Promise(res => setTimeout(res, 4));
      }
      gl.deleteSync(s);
      if (gl.isContextLost()) throw new Error("การ์ดจอหยุดทำงานระหว่างประมวลผล");
    }
  }
  return { Net };
})();

//! Offsets of free-form curves and chains that a single linked spline cannot follow.
//!
//! Where the distance is larger than a bend's radius, the plain normal offset turns back
//! on itself and makes a small loop; distant parts of a curve can cross the same way. A
//! chain of curves (an imported SVG outline, say) also has corners. The offset here is the
//! parallel boundary: the plain offset of every piece, joined round each corner by an arc
//! about the corner, keeping only the points that no point of the source is nearer to than
//! the offset distance. The kept arcs meet at the plain offset's own crossings, where they
//! form sharp corners. Each kept run of pieces is fitted with an open spline and certified
//! like a linked offset; a corner's arc is exact. Every pair of fitted spans is proven
//! apart except at the nodes they share. The result is linked to its sources: the same
//! trimming is run again when they or the distance change, and the curves it made take the
//! new shape as long as the outline still has the same curves in the same order.
use super::{Certificate, H, MAX_ASSOCIATIONS, MAX_OFFSET_WORK, MAX_PIECES, OFFSET_TOLERANCE, Result, bad, certify, normal_point};
use crate::{CurveKind, Sketch, fit_beziers};
use solidcore_record::{Seg2, Vec2};
use solidcraft_geom::prelude::*;

/// Samples of the plain offset used to find its crossings (whole chain).
const MAX_SAMPLES: usize = 32_768;
const MAX_CROSSINGS: usize = 256;
/// Pairs compared while searching for crossings and while proving the result simple.
const MAX_PAIR_WORK: usize = 4_000_000;
/// Source parameter below which a piece of a kept arc is too short to fit on its own.
const SLIVER: f64 = 1e-6;
/// The widest source gap between consecutive chained curves.
const SNAP: f64 = OFFSET_TOLERANCE / 8.0;
/// How far a source end whose parameter speed vanishes (a zero-length Bézier handle) may
/// bulge from the straight line that replaces it. Such an end has infinite curvature and
/// no certifiable normal, so its tail is replaced by its chord; the offset is then that of
/// a source at most `TAIL` from the given one, and fits are certified to what is left.
const TAIL: f64 = OFFSET_TOLERANCE / 4.0;
const FIT_TOLERANCE: f64 = OFFSET_TOLERANCE - TAIL;
/// Joints whose two offset ends are at most this far apart (a kink of a fraction of a
/// degree, common where drawing programs join Bézier curves) are joined directly instead
/// of round a corner arc. The shared node sits midway, so each side's certificate carries
/// half the step.
const KINK: f64 = OFFSET_TOLERANCE * 0.8;

/// What an offset of a free-form curve produced.
#[derive(Clone, Debug, PartialEq)]
pub enum FreeformOffsetOutput {
    /// One spline that follows its source (an associative offset).
    Linked(usize),
    /// Open splines (and corner arcs) meeting where the plain offset's loops were trimmed
    /// away. They follow the source too, as long as the same loops are there to trim.
    Trimmed(Vec<usize>),
}

/// One piece of the plain offset.
#[derive(Clone, Debug)]
enum Src {
    /// An exact source piece, offset along its normal.
    Seg(Seg2),
    /// A corner of the source: the offset turns about `at` on a circle of radius `|d|`,
    /// from the displacement `from` through `sweep` radians (backwards at an inner corner,
    /// where trimming removes it again).
    Corner { at: Vec2, from: Vec2, sweep: f64 },
}

fn rotate(v: Vec2, angle: f64) -> Vec2 {
    v * angle.cos() + v.perp() * angle.sin()
}

/// The plain offset of a chain, by a global parameter `u = piece index + local parameter`.
struct Raw {
    pieces: Vec<Src>,
    d: f64,
    closed: bool,
}

impl Raw {
    fn span(&self) -> f64 {
        self.pieces.len() as f64
    }
    fn locate(&self, u: f64) -> Result<(usize, f64)> {
        if !u.is_finite() || self.pieces.is_empty() {
            return Err(bad("offset parameter out of range"));
        }
        let n = self.span();
        let u = if self.closed { u.rem_euclid(n) } else { u.clamp(0.0, n) };
        let i = (u.floor() as usize).min(self.pieces.len() - 1);
        Ok((i, (u - i as f64).clamp(0.0, 1.0)))
    }
    fn piece(&self, i: usize) -> Result<&Src> {
        self.pieces.get(i).ok_or_else(|| bad("offset piece out of range"))
    }
    fn piece_point(&self, i: usize, t: f64) -> Result<Vec2> {
        match self.piece(i)? {
            Src::Seg(seg) => normal_point(seg, t, self.d),
            Src::Corner { at, from, sweep } => Ok(*at + rotate(*from, sweep * t)),
        }
    }
    fn point(&self, u: f64) -> Result<Vec2> {
        let (i, t) = self.locate(u)?;
        self.piece_point(i, t)
    }
    fn source(&self, u: f64) -> Result<Vec2> {
        let (i, t) = self.locate(u)?;
        Ok(match self.piece(i)? {
            Src::Seg(seg) => seg.point_at(t),
            Src::Corner { at, .. } => *at,
        })
    }
    /// Source velocity and signed curvature (`None` on a corner).
    fn local(&self, u: f64) -> Result<Option<(Vec2, f64)>> {
        let (i, t) = self.locate(u)?;
        let Src::Seg(seg) = self.piece(i)? else { return Ok(None) };
        let d = seg.derivatives(t, 2).ok_or_else(|| bad("nonfinite offset derivative"))?;
        let (v, a) = (d.first().copied().unwrap_or_default(), d.get(1).copied().unwrap_or_default());
        let speed = v.len();
        if !(speed > 1e-12) || !speed.is_finite() {
            return Err(bad("offset source has a stationary tangent"));
        }
        Ok(Some((v, v.cross(a) / speed.powi(3))))
    }
    /// Velocity of the plain offset. On a piece it is the source velocity scaled by
    /// `1 - d·k`, negative where the offset runs backwards inside a fold.
    fn velocity(&self, u: f64) -> Result<Vec2> {
        if let Some((v, k)) = self.local(u)? {
            return Ok(v * (1.0 - self.d * k));
        }
        let (i, t) = self.locate(u)?;
        match self.piece(i)? {
            Src::Corner { from, sweep, .. } => Ok(rotate(*from, sweep * t).perp() * *sweep),
            Src::Seg(_) => Err(bad("offset piece changed kind")),
        }
    }
    /// Cyclic (closed) or plain distance between two parameters.
    fn gap(&self, a: f64, b: f64) -> f64 {
        let g = (a - b).abs();
        if self.closed { g.min(self.span() - g) } else { g }
    }

    /// The plain offset of the oriented curves of a chain.
    fn chain(sk: &Sketch, order: &[(usize, bool)], d: f64) -> Result<Self> {
        if order.is_empty() || !d.is_finite() || d.abs() < 1e-9 {
            return Err(bad("invalid offset"));
        }
        let mut segs = Vec::new();
        for (curve, forward) in order {
            // Lines and arcs of the chain come from their analytic form, free-form curves exactly.
            let mut pieces = sk.exact_pieces(*curve);
            if pieces.is_empty() {
                return Err(bad("offset source has a curve with no exact pieces"));
            }
            if !forward {
                pieces.reverse();
                pieces = pieces.into_iter().map(|p| p.reversed()).collect();
            }
            for piece in pieces {
                segs.extend(exact_pieces(piece)?);
            }
            if segs.len() > MAX_PIECES {
                return Err(bad("offset source has too many pieces"));
            }
        }
        // The joints as given, before any stationary tail is replaced.
        let ends: Vec<(Vec2, Vec2)> = segs.iter().map(|s: &Seg2| (s.point_at(0.0), s.point_at(1.0))).collect();
        let segs = segs.into_iter().map(resolve_stationary_ends).collect::<Result<Vec<_>>>()?;
        let (Some(first), Some(last)) = (ends.first(), ends.last()) else {
            return Err(bad("offset source has no pieces"));
        };
        let closed = first.0.dist(last.1) <= SNAP;
        let mut pieces = Vec::with_capacity(segs.len() * 2);
        for (i, seg) in segs.iter().enumerate() {
            pieces.push(Src::Seg(*seg));
            let j = if i + 1 < segs.len() {
                i + 1
            } else if closed {
                0
            } else {
                continue;
            };
            let (Some(next), Some(joint), Some(next_start)) = (segs.get(j), ends.get(i).map(|e| e.1), ends.get(j).map(|e| e.0)) else { continue };
            if joint.dist(next_start) > SNAP {
                return Err(bad("the offset chain has a gap between consecutive curves"));
            }
            // Replaced tails and any gap are bridged by straight lines through the joints
            // as given, so the chain stays exactly continuous.
            let mut path = vec![seg.point_at(1.0)];
            for p in [joint, next_start, next.point_at(0.0)] {
                // A gap shorter than sketch coincidence (a few 1e-12 between two curves of a saved
                // design) is no piece: a line that short has no tangent (#11597).
                if path.last().is_some_and(|q: &Vec2| q.dist(p) > 1e-9) {
                    path.push(p);
                }
            }
            let mut previous = *seg;
            for w in path.windows(2) {
                let line = Seg2::Line { a: w[0], b: w[1] };
                pieces.extend(corner(&previous, &line, d)?);
                pieces.push(Src::Seg(line));
                previous = line;
            }
            pieces.extend(corner(&previous, next, d)?);
        }
        Ok(Self { pieces, d, closed })
    }
}

/// Source pieces the certificate understands: arcs become exact rational quadratics of at
/// most a quarter turn each; everything else is kept as it is.
fn exact_pieces(seg: Seg2) -> Result<Vec<Seg2>> {
    match seg {
        Seg2::Arc { center, radius, start, sweep } => {
            if !(radius > 1e-9) || !sweep.is_finite() || !(sweep.abs() > 1e-12) {
                return Err(bad("degenerate arc in offset source"));
            }
            let parts = (sweep.abs() / std::f64::consts::FRAC_PI_2).ceil().max(1.0) as usize;
            let step = sweep / parts as f64;
            Ok((0..parts)
                .map(|k| {
                    let a0 = start + step * k as f64;
                    let half = step / 2.0;
                    Seg2::Conic {
                        a: center + Vec2::from_angle(a0) * radius,
                        apex: center + Vec2::from_angle(a0 + half) * (radius / half.cos()),
                        b: center + Vec2::from_angle(a0 + step) * radius,
                        w: half.cos(),
                    }
                })
                .collect())
        }
        // A line shorter than sketch coincidence is the gap between two pieces that meet (a few 1e-12
        // apart in a saved design), not a piece: it has no tangent (#11597).
        Seg2::Line { a, b } if a.dist(b) < 1e-9 => Ok(Vec::new()),
        other => Ok(vec![other]),
    }
}

/// A piece whose parameter speed vanishes at an end (a zero-length Bézier handle) has a
/// well-defined tangent there but infinite curvature and no certifiable normal. The longest
/// end tail that bulges at most `TAIL` from its chord is cut off; the chain bridges it with
/// that chord.
fn resolve_stationary_ends(seg: Seg2) -> Result<Seg2> {
    let scale = seg.length();
    if !(scale > 1e-9) || !scale.is_finite() {
        return Err(bad("offset source has a degenerate piece"));
    }
    let mut seg = seg;
    for at_end in [false, true] {
        let end = if at_end { 1.0 } else { 0.0 };
        if seg.derivative(end).len() > 1e-6 * scale {
            continue;
        }
        let at = |eps: f64| if at_end { 1.0 - eps } else { eps };
        // How far the tail of length `eps` strays from its chord (it grows with `eps`).
        let bulge = |eps: f64| {
            let (p, q) = (seg.point_at(at(eps)), seg.point_at(end));
            let chord = q - p;
            (1..32)
                .map(|k| {
                    let r = seg.point_at(at(eps * k as f64 / 32.0)) - p;
                    let along = (r.dot(chord) / chord.len2().max(1e-300)).clamp(0.0, 1.0);
                    (r - chord * along).len()
                })
                .fold(0.0_f64, f64::max)
        };
        let (mut lo, mut hi) = (0.0_f64, 0.25_f64);
        if bulge(hi) <= TAIL {
            lo = hi;
        } else {
            for _ in 0..60 {
                let mid = (lo + hi) / 2.0;
                if bulge(mid) <= TAIL { lo = mid } else { hi = mid }
            }
        }
        let cut = at(lo);
        if !(lo > 0.0) || !(seg.derivative(cut).len() > 1e-6 * scale) {
            return Err(bad("offset source has a stationary end that cannot be resolved"));
        }
        seg = if at_end { seg.split_at(cut).0 } else { seg.split_at(cut).1 };
    }
    Ok(seg)
}

/// The corner between two consecutive source pieces, unless they meet smoothly enough
/// (their offset ends at most `KINK` apart) to be joined directly.
fn corner(a: &Seg2, b: &Seg2, d: f64) -> Result<Option<Src>> {
    let at = a.point_at(1.0);
    let ta = a.derivative(1.0).normalized().ok_or_else(|| bad("offset source has a stationary tangent"))?;
    let tb = b.derivative(0.0).normalized().ok_or_else(|| bad("offset source has a stationary tangent"))?;
    let sweep = ta.cross(tb).atan2(ta.dot(tb));
    if sweep.abs() > std::f64::consts::PI - 1e-6 {
        return Err(bad("the offset source turns back on itself at a joint"));
    }
    let step = d.abs() * 2.0 * (sweep.abs() / 2.0).sin();
    if step <= KINK {
        return Ok(None);
    }
    Ok(Some(Src::Corner { at, from: ta.perp() * d, sweep }))
}

/// Where the trimmed result could not be made, and why.
enum Trim {
    /// The plain offset never crosses itself: trimming has nothing to remove.
    NothingToTrim,
    Failed(crate::SketchError),
}
impl From<crate::SketchError> for Trim {
    fn from(error: crate::SketchError) -> Self {
        Trim::Failed(error)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum End {
    /// An end of an open chain's offset.
    Free,
    /// Where a closed offset without crossings starts and ends.
    Seam,
    Crossing(usize),
}

struct Crossing {
    u: f64,
    w: f64,
    at: Vec2,
}

/// One kept arc: the global parameters it runs between, and what is at its ends.
struct Arc {
    a: f64,
    b: f64,
    start: End,
    end: End,
}

/// One output curve of a kept arc.
enum Out {
    Spline { nodes: Vec<Vec2>, cubics: Vec<[Vec2; 4]> },
    Arc { center: Vec2, start: Vec2, end: Vec2, sweep: f64 },
}

impl Out {
    fn ends(&self) -> (Vec2, Vec2) {
        match self {
            Out::Spline { nodes, .. } => (nodes.first().copied().unwrap_or_default(), nodes.last().copied().unwrap_or_default()),
            Out::Arc { start, end, .. } => (*start, *end),
        }
    }
}

/// A fitted kept arc: its curves in order, and what is at its ends.
struct Fitted {
    outs: Vec<Out>,
    start: End,
    end: End,
}

fn segment_crossing(a0: Vec2, a1: Vec2, b0: Vec2, b1: Vec2) -> Option<(f64, f64)> {
    let (r, s) = (a1 - a0, b1 - b0);
    let den = r.cross(s);
    if !(den.abs() > 1e-300) {
        return None;
    }
    let q = b0 - a0;
    let (t, w) = (q.cross(s) / den, q.cross(r) / den);
    ((0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&w)).then_some((t, w))
}

/// Newton on `O(u) = O(w)` from a polyline estimate. Tangential contacts are refused: a
/// trimmed corner is only defined where the two branches really cross.
fn refine_crossing(raw: &Raw, mut u: f64, mut w: f64) -> Result<Crossing> {
    let limit = |x: f64| if raw.closed { x } else { x.clamp(0.0, raw.span()) };
    for _ in 0..50 {
        let (p, q) = (raw.point(u)?, raw.point(w)?);
        let f = p - q;
        let (du, dw) = (raw.velocity(u)?, raw.velocity(w)?);
        // Solve [du, -dw]·[x, y] = -f by Cramer's rule.
        let det = du.cross(dw * -1.0);
        if !(det.abs() > 1e-9 * du.len() * dw.len()) || !det.is_finite() {
            return Err(bad("the offset touches itself tangentially; change the distance slightly"));
        }
        let (x, y) = ((f * -1.0).cross(dw * -1.0) / det, du.cross(f * -1.0) / det);
        u = limit(u + x);
        w = limit(w + y);
        if f.len() <= 1e-12 * (1.0 + p.len()) && x.abs().max(y.abs()) < 1e-13 {
            break;
        }
    }
    let (p, q) = (raw.point(u)?, raw.point(w)?);
    if p.dist(q) > 1e-9 * (1.0 + p.len()) || raw.gap(u, w) < 1e-9 {
        return Err(bad("an offset crossing could not be located exactly"));
    }
    Ok(Crossing { u, w, at: p.lerp(q, 0.5) })
}

/// The same crossing, found twice (either way round).
fn same_crossing(raw: &Raw, a: &Crossing, b: &Crossing) -> bool {
    (raw.gap(a.u, b.u) < 1e-7 && raw.gap(a.w, b.w) < 1e-7) || (raw.gap(a.u, b.w) < 1e-7 && raw.gap(a.w, b.u) < 1e-7)
}

/// Every crossing of the plain offset with itself, from a dense polyline (pruned by chunk
/// bounding boxes) refined by Newton.
fn crossings(raw: &Raw, us: &[f64], pts: &[Vec2]) -> Result<Vec<Crossing>> {
    let segments = pts.len().saturating_sub(1);
    const CHUNK: usize = 8;
    let boxes: Vec<(Vec2, Vec2)> = (0..segments.div_ceil(CHUNK))
        .map(|c| {
            let slice = pts.get(c * CHUNK..((c + 1) * CHUNK + 1).min(pts.len())).unwrap_or(&[]);
            bounds(slice)
        })
        .collect();
    let mut found: Vec<Crossing> = Vec::new();
    let mut work = 0usize;
    for (ca, a) in boxes.iter().enumerate() {
        for (cb, b) in boxes.iter().enumerate().skip(ca) {
            work += 1;
            if work > MAX_PAIR_WORK {
                return Err(bad("offset crossing search work limit exceeded"));
            }
            if a.1.x < b.0.x || b.1.x < a.0.x || a.1.y < b.0.y || b.1.y < a.0.y {
                continue;
            }
            for j in ca * CHUNK..((ca + 1) * CHUNK).min(segments) {
                for k in (cb * CHUNK).max(j + 2)..((cb + 1) * CHUNK).min(segments) {
                    if raw.closed && j == 0 && k + 1 == segments {
                        continue;
                    }
                    let (Some(a0), Some(a1), Some(b0), Some(b1)) = (pts.get(j), pts.get(j + 1), pts.get(k), pts.get(k + 1)) else { continue };
                    let Some((s, t)) = segment_crossing(*a0, *a1, *b0, *b1) else { continue };
                    let (Some(uj), Some(uj1), Some(uk), Some(uk1)) = (us.get(j), us.get(j + 1), us.get(k), us.get(k + 1)) else { continue };
                    // Chords either side of a fold's tip, or of a tiny corner arc, can touch
                    // without the offset crossing itself there: Newton then collapses onto
                    // one point or stalls on a tangential contact. Such a candidate is not
                    // a corner to trim at. Skipping a real one is safe: the result is
                    // proven free of crossings before it is accepted.
                    let crossing = match refine_crossing(raw, uj + (uj1 - uj) * s, uk + (uk1 - uk) * t) {
                        Ok(crossing) if raw.gap(crossing.u, crossing.w) >= 1e-4 => crossing,
                        _ => continue,
                    };
                    if !found.iter().any(|c| same_crossing(raw, c, &crossing)) {
                        if found.len() >= MAX_CROSSINGS {
                            return Err(bad("the offset crosses itself too many times"));
                        }
                        found.push(crossing);
                    }
                }
            }
        }
    }
    // At an inner corner the offsets either side always cross close to it; a small corner's
    // overlap can be far finer than the samples, so it is looked for from the corner itself.
    let n = raw.span();
    for (index, piece) in raw.pieces.iter().enumerate() {
        let Src::Corner { sweep, .. } = piece else { continue };
        if !(raw.d * sweep > 0.0) || (!raw.closed && (index == 0 || index + 1 >= raw.pieces.len())) {
            continue;
        }
        let (before, after) = (index as f64, index as f64 + 1.0);
        let near = |u: f64, w: f64| raw.gap(u, before) < 1.0 && raw.gap(w, after) < 1.0;
        if found.iter().any(|c| near(c.u, c.w) || near(c.w, c.u)) {
            continue;
        }
        for eps in [1e-7, 1e-5, 1e-3, 1e-1] {
            let start = if raw.closed { (before - eps).rem_euclid(n) } else { (before - eps).max(0.0) };
            let Ok(crossing) = refine_crossing(raw, start, after + eps) else { continue };
            let (u, w) = (crossing.u.rem_euclid(n), crossing.w.rem_euclid(n));
            // It must lie on the pieces either side of the corner, not on the corner's arc.
            if raw.gap(u, before) < 1.0 && raw.gap(w, after) < 1.0 && raw.gap(u, before) > 0.0 && raw.gap(w, after) > 0.0 {
                let (Ok((pu, _)), Ok((pw, _))) = (raw.locate(u), raw.locate(w)) else { continue };
                if found.iter().any(|c| same_crossing(raw, c, &crossing)) {
                    break;
                }
                if pu != index && pw != index {
                    if found.len() >= MAX_CROSSINGS {
                        return Err(bad("the offset crosses itself too many times"));
                    }
                    found.push(crossing);
                    break;
                }
            }
        }
    }
    Ok(found)
}

/// Is any point of the source nearer to `q` than `|d|`? The sampled source gives the
/// candidates; Newton on each nearby local minimum gives the exact distance.
fn nearer_than_offset(raw: &Raw, us: &[f64], src: &[Vec2], q: Vec2) -> Result<bool> {
    // A kept point's nearest source point is its own foot, exactly `|d|` away (to rounding);
    // a point inside a small inner corner is nearer by about `|d|·θ²/8`, which for any
    // corner not joined directly is far above this slack.
    let limit = raw.d.abs() * (1.0 - 1e-9) - 1e-12;
    let dist: Vec<f64> = src.iter().map(|p| p.dist(q)).collect();
    if dist.iter().any(|d| *d < limit) {
        return Ok(true);
    }
    let spacing = src.windows(2).map(|w| w[0].dist(w[1])).fold(0.0_f64, f64::max);
    let at = |i: usize| dist.get(i).copied().unwrap_or(f64::INFINITY);
    // A corner's samples all stand for its one point, already measured exactly above; only
    // samples on source pieces (a piece's first sample included) are refined.
    let on_piece = |i: usize| us.get(i).and_then(|u| raw.locate(*u).ok()).is_some_and(|(p, _)| matches!(raw.pieces.get(p), Some(Src::Seg(_))));
    let mut candidates: Vec<usize> = (0..dist.len())
        .filter(|i| {
            let before = if *i == 0 { f64::INFINITY } else { at(i - 1) };
            on_piece(*i) && at(*i) <= before && at(*i) <= at(i + 1) && at(*i) < raw.d.abs() + spacing
        })
        .collect();
    candidates.sort_by(|a, b| at(*a).total_cmp(&at(*b)));
    candidates.truncate(64);
    for i in candidates {
        let Some(&u0) = us.get(i) else { continue };
        let (piece, t0) = raw.locate(u0)?;
        // A sample on a piece boundary is also the end of the previous piece.
        let mut starts = vec![(piece, t0)];
        if t0 == 0.0 {
            if piece > 0 {
                starts.push((piece - 1, 1.0));
            } else if raw.closed {
                starts.push((raw.pieces.len() - 1, 1.0));
            }
        }
        for (piece, mut t) in starts {
            // A corner's source is its single point, already sampled exactly.
            let Some(Src::Seg(seg)) = raw.pieces.get(piece) else { continue };
            for _ in 0..40 {
                let d = seg.derivatives(t, 2).ok_or_else(|| bad("nonfinite offset derivative"))?;
                let (v, a) = (d.first().copied().unwrap_or_default(), d.get(1).copied().unwrap_or_default());
                let h = seg.point_at(t) - q;
                let (f, df) = (h.dot(v), v.dot(v) + h.dot(a));
                if !(df.abs() > 1e-300) {
                    break;
                }
                let next = (t - f / df).clamp(0.0, 1.0);
                if (next - t).abs() < 1e-15 {
                    t = next;
                    break;
                }
                t = next;
            }
            if seg.point_at(t).dist(q) < limit {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Where to test whether the stretch `a..b` of the plain offset is kept: the middle of its
/// longest stretch on an offset piece (a corner's arc can be a sliver at an inner corner).
fn test_parameter(raw: &Raw, a: f64, b: f64) -> f64 {
    let mut best = ((a + b) / 2.0, f64::NEG_INFINITY);
    let mut u = a;
    let mut guard = 0;
    while u < b && guard <= raw.pieces.len() + 1 {
        guard += 1;
        let next = (u.floor() + 1.0).min(b);
        let on_piece = raw.locate((u + next) / 2.0).is_ok_and(|(i, _)| matches!(raw.pieces.get(i), Some(Src::Seg(_))));
        if on_piece && next - u > best.1 {
            best = ((u + next) / 2.0, next - u);
        }
        u = next;
    }
    best.0
}

/// The kept arcs of the plain offset, between consecutive crossings.
fn kept_arcs(raw: &Raw, crossings: &[Crossing], us: &[f64], src: &[Vec2]) -> Result<Vec<Arc>> {
    let n = raw.span();
    let norm = |u: f64| if raw.closed { u.rem_euclid(n) } else { u };
    let mut cuts: Vec<(f64, End)> =
        crossings.iter().enumerate().flat_map(|(i, c)| [(norm(c.u), End::Crossing(i)), (norm(c.w), End::Crossing(i))]).collect();
    cuts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut bounds = Vec::new();
    if raw.closed {
        match cuts.first().copied() {
            None => bounds.push(((0.0, End::Seam), (n, End::Seam))),
            Some(first) => {
                for (i, cut) in cuts.iter().enumerate() {
                    let next = cuts.get(i + 1).copied().unwrap_or((first.0 + n, first.1));
                    bounds.push((*cut, next));
                }
            }
        }
    } else {
        let mut all = vec![(0.0, End::Free)];
        all.extend(cuts);
        all.push((n, End::Free));
        bounds.extend(all.windows(2).map(|w| (w[0], w[1])));
    }
    let mut arcs = Vec::new();
    for ((a, start), (b, end)) in bounds {
        if b - a < 1e-9 {
            return Err(bad("the offset crosses itself at an ambiguous point; change the distance slightly"));
        }
        if !nearer_than_offset(raw, us, src, raw.point(test_parameter(raw, a, b))?)? {
            arcs.push(Arc { a, b, start, end });
        }
    }
    if arcs.is_empty() {
        return Err(bad("the offset distance leaves nothing on this side of the curve"));
    }
    for i in 0..crossings.len() {
        let ends = arcs.iter().map(|arc| usize::from(arc.start == End::Crossing(i)) + usize::from(arc.end == End::Crossing(i))).sum::<usize>();
        if ends > 2 {
            return Err(bad("the offset crosses itself at an ambiguous point; change the distance slightly"));
        }
    }
    Ok(arcs)
}

/// The plain-offset pieces a kept arc runs over: `(piece, from, to)` in local parameters.
fn arc_pieces(raw: &Raw, arc: &Arc) -> Result<Vec<(usize, f64, f64)>> {
    let mut out = Vec::new();
    let mut u = arc.a;
    while u < arc.b - 1e-12 {
        if out.len() > raw.pieces.len() + 1 {
            return Err(bad("offset arc walk did not terminate"));
        }
        let base = u.floor();
        let next = (base + 1.0).min(arc.b);
        let piece = (base as i64).rem_euclid(raw.pieces.len() as i64) as usize;
        out.push((piece, u - base, next - base));
        u = next;
    }
    // A sliver of a piece at either end is covered by its neighbour's span when both are
    // offset pieces; its end node is still placed exactly on the crossing.
    let seg = |p: Option<&(usize, f64, f64)>| p.is_some_and(|p| matches!(raw.pieces.get(p.0), Some(Src::Seg(_))));
    if out.len() > 1 && out.first().is_some_and(|p| p.2 - p.1 < SLIVER) && seg(out.first()) && seg(out.get(1)) {
        out.remove(0);
    }
    let len = out.len();
    if len > 1 && out.last().is_some_and(|p| p.2 - p.1 < SLIVER) && seg(out.last()) && seg(out.get(len - 2)) {
        out.pop();
    }
    if out.is_empty() || out.iter().any(|p| !(p.2 > p.1)) {
        return Err(bad("offset arc is too short to fit"));
    }
    Ok(out)
}

/// Fit and certify one run of offset pieces. It starts from a few uniform spans per piece
/// and halves only the spans whose certificate fails, refitting until every span passes:
/// sharp bends (a fold's neighbourhood, a set-aside tail) get dense nodes, the rest stays
/// sparse. `first` and `last` are its exact end nodes. `floor` is the fewest nodes the
/// spline may have: a rebuilt offset never has fewer than the spline it replaces, so the
/// points that spline owns all stay in use.
fn fit_run(raw: &Raw, run: &[(usize, f64, f64)], first: Vec2, last: Vec2, floor: usize, work: &mut usize) -> Result<Out> {
    let segs = run
        .iter()
        .map(|(i, _, _)| match raw.piece(*i)? {
            Src::Seg(seg) => Ok(*seg),
            Src::Corner { .. } => Err(bad("offset run changed kind")),
        })
        .collect::<Result<Vec<_>>>()?;
    let homogeneous = segs.iter().map(H::from_seg).collect::<Result<Vec<_>>>()?;
    // Spans as (piece of the run, from, to) in that piece's parameter.
    let mut spans: Vec<(usize, f64, f64)> = run
        .iter()
        .enumerate()
        .flat_map(|(k, (_, from, to))| (0..2).map(move |j| (k, from + (to - from) * j as f64 / 2.0, from + (to - from) * (j + 1) as f64 / 2.0)))
        .collect();
    let mut failure = bad("offset arc could not be fitted");
    for _ in 0..32 {
        if spans.len() + 1 >= crate::MAX_SPLINE_POINTS {
            return Err(bad(&format!("the offset needs too many spline nodes to stay within {OFFSET_TOLERANCE} mm: {failure}")));
        }
        let mut nodes = vec![first];
        let mut ranges = Vec::with_capacity(spans.len());
        for (index, (k, a, b)) in spans.iter().enumerate() {
            let (Some(h), Some(seg)) = (homogeneous.get(*k), segs.get(*k)) else { return Err(bad("offset span out of range")) };
            ranges.push(h.range(*a, *b));
            let mut node = normal_point(seg, *b, raw.d)?;
            // A joint joined directly: the node sits midway between the two pieces' ends.
            if let Some((next, start_at, _)) = spans.get(index + 1)
                && next != k
                && let Some(following) = segs.get(*next)
            {
                node = node.lerp(normal_point(following, *start_at, raw.d)?, 0.5);
            }
            nodes.push(node);
        }
        if let Some(end) = nodes.last_mut() {
            *end = last;
        }
        let cubics = fit_beziers(&nodes);
        if cubics.len() != ranges.len() {
            return Err(bad("offset arc fit has a mismatched span count"));
        }
        let mut failing = Vec::new();
        let mut near = first;
        for (index, (h, c)) in ranges.iter().zip(&cubics).enumerate() {
            // Each span's certificate stands on its own; either correspondence proves it.
            let proven = [false, true].into_iter().any(|align_midpoints| {
                match certify(h, c, raw.d, Certificate { tolerance: FIT_TOLERANCE, align_midpoints }, 0, work) {
                    Ok(()) => true,
                    Err(error) => {
                        failure = error;
                        false
                    }
                }
            });
            if !proven {
                failing.push(index);
                near = c[0];
            }
            if *work == 0 {
                return Err(bad("offset certificate work limit exceeded"));
            }
        }
        if failing.is_empty() {
            if nodes.len() >= floor {
                return Ok(Out::Spline { nodes, cubics });
            }
            // Too few nodes for the spline this replaces: halve the widest spans.
            let mut widest: Vec<usize> = (0..spans.len()).collect();
            widest.sort_by(|a, b| {
                let width = |i: &usize| spans.get(*i).map_or(0.0, |s| s.2 - s.1);
                width(b).total_cmp(&width(a))
            });
            widest.truncate(floor - nodes.len());
            widest.sort_unstable();
            failing = widest;
        }
        let located = |error: &crate::SketchError| {
            bad(&format!("the offset could not be fitted within {OFFSET_TOLERANCE} mm near ({:.3}, {:.3}): {error}", near.x, near.y))
        };
        let mut refined = Vec::with_capacity(spans.len() + failing.len());
        for (index, (k, a, b)) in spans.iter().copied().enumerate() {
            if failing.binary_search(&index).is_ok() {
                let mid = (a + b) / 2.0;
                if !(mid > a && mid < b) {
                    return Err(located(&failure));
                }
                refined.push((k, a, mid));
                refined.push((k, mid, b));
            } else {
                refined.push((k, a, b));
            }
        }
        spans = refined;
    }
    Err(failure)
}

/// Fit one kept arc: a spline for each run of offset pieces, an exact arc for each corner.
/// `floors` is the fewest nodes each output curve may have, by its place among all the
/// offset's curves; `placed` counts the curves fitted so far.
fn fit_arc(raw: &Raw, arc: &Arc, crossings: &[Crossing], floors: &[usize], placed: &mut usize, work: &mut usize) -> Result<Fitted> {
    let pieces = arc_pieces(raw, arc)?;
    let at = |end: End, u: f64| match end {
        End::Crossing(i) => crossings.get(i).map(|c| c.at).ok_or_else(|| bad("offset crossing vanished")),
        End::Free | End::Seam => raw.point(u),
    };
    let (first, last) = (at(arc.start, arc.a)?, at(arc.end, arc.b)?);
    let is_corner = |p: &(usize, f64, f64)| matches!(raw.pieces.get(p.0), Some(Src::Corner { .. }));
    let mut outs = Vec::new();
    let mut start = 0;
    let mut node = first;
    while start < pieces.len() {
        let corner = pieces.get(start).is_some_and(is_corner);
        let mut end = start + 1;
        while end < pieces.len() && !corner && !pieces.get(end).is_some_and(is_corner) {
            end += 1;
        }
        let run = pieces.get(start..end).ok_or_else(|| bad("offset run out of range"))?;
        let (piece, from, to) = run.last().copied().ok_or_else(|| bad("empty offset run"))?;
        // Runs share their boundary node: the start of the following run, which for a
        // corner lies exactly on its arc.
        let next = match pieces.get(end) {
            None => last,
            Some((following, start_at, _)) => raw.piece_point(*following, *start_at)?,
        };
        if corner {
            let Src::Corner { at, sweep, .. } = raw.piece(piece)? else { return Err(bad("offset run changed kind")) };
            outs.push(Out::Arc { center: *at, start: node, end: next, sweep: sweep * (to - from) });
        } else {
            outs.push(fit_run(raw, run, node, next, floors.get(*placed).copied().unwrap_or(0), work)?);
        }
        *placed += 1;
        node = next;
        start = end;
    }
    Ok(Fitted { outs, start: arc.start, end: arc.end })
}

fn bounds(points: &[Vec2]) -> (Vec2, Vec2) {
    let lo = points.iter().fold(Vec2::new(f64::INFINITY, f64::INFINITY), |m, p| Vec2::new(m.x.min(p.x), m.y.min(p.y)));
    let hi = points.iter().fold(Vec2::new(f64::NEG_INFINITY, f64::NEG_INFINITY), |m, p| Vec2::new(m.x.max(p.x), m.y.max(p.y)));
    (lo, hi)
}

/// A piece of the result for the simplicity proof, with a convex hull that contains it.
#[derive(Clone, Copy)]
enum Prim {
    Cubic([Vec2; 4]),
    /// An arc of at most a quarter turn with its exact (shared) end nodes.
    Arc {
        center: Vec2,
        radius: f64,
        start_angle: f64,
        sweep: f64,
        p0: Vec2,
        p1: Vec2,
    },
}

impl Prim {
    fn hull(&self) -> Vec<Vec2> {
        match *self {
            Prim::Cubic(c) => c.to_vec(),
            Prim::Arc { center, radius, start_angle, sweep, p0, p1 } => {
                // The tangents at the ends meet at the apex; a little slack covers the
                // nodes' rounding off the exact circle.
                let half = sweep / 2.0;
                let apex = center + Vec2::from_angle(start_angle + half) * (radius / half.cos() * (1.0 + 1e-9) + 1e-9);
                vec![p0, apex, p1]
            }
        }
    }
    fn halves(&self) -> (Prim, Prim) {
        match *self {
            Prim::Cubic(c) => {
                let (ab, bc, cd) = (c[0].lerp(c[1], 0.5), c[1].lerp(c[2], 0.5), c[2].lerp(c[3], 0.5));
                let (abc, bcd) = (ab.lerp(bc, 0.5), bc.lerp(cd, 0.5));
                let m = abc.lerp(bcd, 0.5);
                (Prim::Cubic([c[0], ab, abc, m]), Prim::Cubic([m, bcd, cd, c[3]]))
            }
            Prim::Arc { center, radius, start_angle, sweep, p0, p1 } => {
                let half = sweep / 2.0;
                let mid = center + Vec2::from_angle(start_angle + half) * radius;
                (
                    Prim::Arc { center, radius, start_angle, sweep: half, p0, p1: mid },
                    Prim::Arc { center, radius, start_angle: start_angle + half, sweep: half, p0: mid, p1 },
                )
            }
        }
    }
    fn locally_simple(&self) -> bool {
        match self {
            Prim::Cubic(c) => {
                let chord = c[3] - c[0];
                c.windows(2).all(|p| (p[1] - p[0]).dot(chord) > 1e-18)
            }
            Prim::Arc { sweep, .. } => sweep.abs() <= std::f64::consts::FRAC_PI_2 + 1e-12,
        }
    }
}

/// Do the convex hulls of two pieces meet nowhere except (when given) at the one node
/// they share? Proven by bounding boxes, separating axes, and a separating line through
/// the shared node, subdividing where those are not yet decisive.
fn apart(a: &Prim, b: &Prim, depth: u8, work: &mut usize) -> bool {
    if *work == 0 {
        return false;
    }
    *work -= 1;
    let (ha, hb) = (a.hull(), b.hull());
    let (Some(&a0), Some(&a1), Some(&b0), Some(&b1)) = (ha.first(), ha.last(), hb.first(), hb.last()) else { return false };
    let shared = [a0, a1].into_iter().find(|p| *p == b0 || *p == b1);
    let (la, ua) = bounds(&ha);
    let (lb, ub) = bounds(&hb);
    if shared.is_none() && (ua.x < lb.x || ub.x < la.x || ua.y < lb.y || ub.y < la.y) {
        return true;
    }
    let apart_along = |n: Vec2| {
        let range = |h: &[Vec2]| h.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| (lo.min(p.dot(n)), hi.max(p.dot(n))));
        let ((alo, ahi), (blo, bhi)) = (range(&ha), range(&hb));
        ahi < blo || bhi < alo
    };
    match shared {
        None => {
            if [a1 - a0, b1 - b0].iter().any(|d| apart_along(d.perp())) {
                return true;
            }
        }
        Some(x) => {
            let others = |h: &[Vec2]| h.iter().copied().filter(|p| *p != x).collect::<Vec<_>>();
            let (oa, ob) = (others(&ha), others(&hb));
            // Both ends at the shared node means the pieces close a loop on themselves.
            if oa.len() + 1 < ha.len() || ob.len() + 1 < hb.len() || oa.is_empty() || ob.is_empty() {
                return false;
            }
            let leg = |o: &[Vec2]| o.iter().fold(Vec2::ZERO, |s, p| s + (*p - x)).normalized();
            let near = |h: &[Vec2]| {
                let p = if h.first() == Some(&x) { h.get(1) } else { h.len().checked_sub(2).and_then(|i| h.get(i)) };
                p.and_then(|p| (*p - x).normalized())
            };
            let candidates = [leg(&ob).zip(leg(&oa)).map(|(b, a)| b - a), near(&hb).zip(near(&ha)).map(|(b, a)| b - a)];
            let separates = |n: Vec2| {
                n.normalized().is_some_and(|n| {
                    let scale = 1e-12 * (1.0 + x.len());
                    oa.iter().all(|p| (*p - x).dot(n) < -scale) && ob.iter().all(|p| (*p - x).dot(n) > scale)
                })
            };
            if candidates.into_iter().flatten().any(separates) {
                return true;
            }
        }
    }
    if depth >= 14 {
        return false;
    }
    // Split the larger piece; a half that no longer holds the shared node is compared plainly.
    if (ua - la).len() >= (ub - lb).len() {
        let (a0, a1) = a.halves();
        apart(&a0, b, depth + 1, work) && apart(&a1, b, depth + 1, work)
    } else {
        let (b0, b1) = b.halves();
        apart(a, &b0, depth + 1, work) && apart(a, &b1, depth + 1, work)
    }
}

/// Every piece locally simple, and no two pieces of the result meeting anywhere but at a
/// node they share.
fn verify_trimmed(fitted: &[Fitted]) -> Result<()> {
    let mut prims = Vec::new();
    for out in fitted.iter().flat_map(|f| f.outs.iter()) {
        match out {
            Out::Spline { cubics, .. } => prims.extend(cubics.iter().map(|c| Prim::Cubic(*c))),
            Out::Arc { center, start, end, sweep } => {
                let mut pending = vec![Prim::Arc {
                    center: *center,
                    radius: center.dist(*start),
                    start_angle: (*start - *center).angle(),
                    sweep: *sweep,
                    p0: *start,
                    p1: *end,
                }];
                while let Some(prim) = pending.pop() {
                    match prim {
                        Prim::Arc { sweep, .. } if sweep.abs() > std::f64::consts::FRAC_PI_4 => {
                            let (a, b) = prim.halves();
                            pending.push(b);
                            pending.push(a);
                        }
                        _ => prims.push(prim),
                    }
                }
            }
        }
    }
    if prims.iter().any(|p| !p.locally_simple()) {
        return Err(bad("trimmed offset spline cannot be certified locally simple"));
    }
    let mut work = MAX_PAIR_WORK;
    for i in 0..prims.len() {
        for j in i + 1..prims.len() {
            let (Some(a), Some(b)) = (prims.get(i), prims.get(j)) else { continue };
            if !apart(a, b, 0, &mut work) {
                return Err(bad(if work == 0 {
                    "trimmed offset certificate work limit exceeded"
                } else {
                    "trimmed offset self-intersection or ambiguous close contact"
                }));
            }
        }
    }
    Ok(())
}

/// The tightest bend on the offset side (radius and the source point there), if the
/// distance is larger than it.
fn tightest_fold(raw: &Raw, us: &[f64]) -> Option<(f64, Vec2)> {
    let mut worst: Option<(f64, Vec2)> = None;
    for u in us {
        let Ok(Some((_, k))) = raw.local(*u) else { continue };
        if raw.d * k >= 1.0 {
            let radius = 1.0 / k.abs();
            if worst.is_none_or(|(r, _)| radius < r)
                && let Ok(p) = raw.source(*u)
            {
                worst = Some((radius, p));
            }
        }
    }
    worst
}

/// Parameters at which the plain offset is sampled: uniform, refined wherever the offset
/// turns sharply or reverses between two samples, so a small fold (near a zero-length
/// handle, say) is not stepped over.
fn samples(raw: &Raw) -> Vec<f64> {
    let per = (MAX_SAMPLES / 4 / raw.pieces.len().max(1)).clamp(16, 512);
    let mut out = Vec::new();
    let direction = |u: f64| raw.velocity(u).ok().and_then(Vec2::normalized);
    let mut budget = MAX_SAMPLES;
    for j in 0..raw.pieces.len() * per {
        let (a, b) = (j as f64 / per as f64, (j + 1) as f64 / per as f64);
        out.push(a);
        // Depth-first refinement of [a, b], emitting interior samples in order.
        let mut stack = vec![(a, b, 0u8)];
        while let Some((lo, hi, depth)) = stack.pop() {
            let turn = match (direction(lo), direction(hi)) {
                (Some(p), Some(q)) => p.cross(q).atan2(p.dot(q)).abs(),
                _ => f64::INFINITY,
            };
            if turn <= 0.35 || depth >= 16 || budget == 0 {
                if lo > a {
                    out.push(lo);
                }
                continue;
            }
            budget -= 1;
            let mid = (lo + hi) / 2.0;
            stack.push((mid, hi, depth + 1));
            stack.push((lo, mid, depth + 1));
        }
    }
    out.push(raw.span());
    out.dedup();
    out
}

/// An error that names the bend the offset folds over, when it does.
fn explain(raw: &Raw, us: &[f64], distance: f64, error: crate::SketchError) -> crate::SketchError {
    match tightest_fold(raw, us) {
        Some((radius, at)) => bad(&format!(
            "the offset distance {:.4} mm is larger than the tightest bend on that side (radius {radius:.4} mm near ({:.3}, {:.3})), and the offset could not be trimmed there: {error}",
            distance.abs(),
            at.x,
            at.y
        )),
        None => error,
    }
}

/// The trimmed offset's kept arcs, fitted and proven apart, and how many crossings its plain
/// offset has. `floors` is the fewest nodes each output curve may have, in order.
fn trimmed(raw: &Raw, us: &[f64], require_crossing: bool, floors: &[usize]) -> std::result::Result<(Vec<Fitted>, usize), Trim> {
    let pts = us.iter().map(|u| raw.point(*u)).collect::<Result<Vec<_>>>()?;
    let src = us.iter().map(|u| raw.source(*u)).collect::<Result<Vec<_>>>()?;
    let crossings = crossings(raw, us, &pts)?;
    if crossings.is_empty() && require_crossing {
        return Err(Trim::NothingToTrim);
    }
    let arcs = kept_arcs(raw, &crossings, us, &src)?;
    let mut work = MAX_OFFSET_WORK;
    let mut placed = 0;
    let fitted = arcs.iter().map(|arc| fit_arc(raw, arc, &crossings, floors, &mut placed, &mut work)).collect::<Result<Vec<_>>>()?;
    verify_trimmed(&fitted)?;
    Ok((fitted, crossings.len()))
}

impl Sketch {
    /// Offset a free-form curve. A curve whose plain offset is smooth and simple gets a
    /// linked offset ([`Sketch::add_freeform_offset`]). Where the plain offset folds over
    /// a bend tighter than the distance, or crosses itself, its loops are trimmed away and
    /// the result is open splines meeting at sharp corners, linked like a chain's offset.
    pub fn offset_freeform(&mut self, source: usize, forward: bool, distance: f64) -> Result<FreeformOffsetOutput> {
        // An offset of a trimmed offset's curve is trimmed too, crossing or not: the two are
        // then rebuilt in turn, which a linked offset of a spline that gains nodes cannot be.
        let derived = self.curves.get(source).is_some_and(|c| self.follows_trimmed_offset(&c.id));
        let plain = if derived {
            None
        } else {
            match self.add_freeform_offset(source, forward, distance) {
                Ok(output) => return Ok(FreeformOffsetOutput::Linked(output)),
                Err(error) => Some(error),
            }
        };
        let raw = match (Raw::chain(self, &[(source, forward)], distance), &plain) {
            (Ok(raw), _) => raw,
            (Err(error), None) => return Err(error),
            (Err(_), Some(plain)) => return Err(plain.clone()),
        };
        let us = samples(&raw);
        if self.offsets.iter().filter(|o| o.trimmed).count() >= MAX_ASSOCIATIONS {
            return Err(bad("too many associative free-form offsets"));
        }
        match self.add_trimmed_offset(&raw, &us, !derived) {
            Ok(curves) => {
                self.link_trimmed_offset(&[(source, forward)], &curves, distance, raw.closed)?;
                Ok(FreeformOffsetOutput::Trimmed(curves))
            }
            Err(Trim::NothingToTrim) => Err(explain(&raw, &us, distance, plain.unwrap_or_else(|| bad("offset produced nothing")))),
            Err(Trim::Failed(error)) => Err(explain(&raw, &us, distance, error)),
        }
    }

    /// Record the association of a trimmed offset just made: `curves` follow `order` at `distance`.
    fn link_trimmed_offset(&mut self, order: &[(usize, bool)], curves: &[usize], distance: f64, closed: bool) -> Result<()> {
        let id = |sk: &Sketch, i: usize| sk.curves.get(i).map(|c| c.id.clone()).ok_or_else(|| bad("offset curve vanished"));
        let sources = order.iter().map(|(i, forward)| Ok((id(self, *i)?, *forward))).collect::<Result<Vec<_>>>()?;
        let outputs = curves.iter().map(|i| id(self, *i)).collect::<Result<Vec<_>>>()?;
        self.offsets.push(crate::OffsetChain { sources, outputs, distance, closed, topology: None, trimmed: true });
        Ok(())
    }

    /// Offset a chain of curves (oriented, in order) that includes free-form curves: every
    /// piece's normal offset, joined round the chain's corners by arcs and trimmed where it
    /// folds or crosses. The result is linked to the chain.
    pub fn offset_freeform_chain(&mut self, order: &[(usize, bool)], distance: f64) -> Result<Vec<usize>> {
        if self.offsets.iter().filter(|o| o.trimmed).count() >= MAX_ASSOCIATIONS {
            return Err(bad("too many associative free-form offsets"));
        }
        let raw = Raw::chain(self, order, distance)?;
        let us = samples(&raw);
        let curves = match self.add_trimmed_offset(&raw, &us, false) {
            Ok(curves) => curves,
            Err(Trim::NothingToTrim) => return Err(bad("offset chain produced nothing")),
            Err(Trim::Failed(error)) => return Err(explain(&raw, &us, distance, error)),
        };
        self.link_trimmed_offset(order, &curves, distance, raw.closed)?;
        Ok(curves)
    }

    /// Validate saved trimmed offsets without evaluating their geometry. Every record names
    /// existing curves; its outputs are the open splines and arcs a trimmed offset is made
    /// of, belong to no other offset, and own their points (no other curve uses one, and
    /// only the ends are shared between them); a record follows only curves that are there
    /// before it is rebuilt, so never its own, a later trimmed offset's, and no other kind
    /// of offset follows a trimmed one.
    pub(crate) fn validate_trimmed_offsets(&self) -> Result<()> {
        if !self.offsets.iter().any(|o| o.trimmed) {
            return Ok(());
        }
        if self.points.len() > 100_000 || self.curves.len() > 100_000 {
            return Err(bad("offset sketch exceeds ownership validation limit"));
        }
        let invalid = || bad("invalid trimmed offset association");
        let id_ok = |id: &String| !id.is_empty() && id.len() <= 512;
        let trimmed: std::collections::BTreeSet<&str> =
            self.offsets.iter().filter(|o| o.trimmed).flat_map(|o| o.outputs.iter().map(String::as_str)).collect();
        if self
            .offsets
            .iter()
            .filter(|o| !o.trimmed)
            .any(|o| o.sources.iter().any(|(id, _)| trimmed.contains(id.as_str())) || trimmed.iter().any(|id| o.owns_output(id)))
            || self.freeform_offsets.iter().any(|o| trimmed.contains(o.source.as_str()) || trimmed.contains(o.output.as_str()))
        {
            return Err(invalid());
        }
        // Which record's outputs use each point, and how many times.
        let mut owner: Vec<Option<usize>> = vec![None; self.points.len()];
        let mut uses = vec![0usize; self.points.len()];
        let mut record_of = std::collections::BTreeMap::new();
        let mut rebuilt = std::collections::BTreeSet::new();
        let mut work = 0usize;
        for (index, record) in self.offsets.iter().filter(|o| o.trimmed).enumerate() {
            if index >= MAX_ASSOCIATIONS
                || record.topology.is_some()
                || record.sources.is_empty()
                || record.sources.len() > 10_000
                || record.outputs.is_empty()
                || record.outputs.len() > 10_000
                || !record.distance.is_finite()
                || record.distance.abs() < 1e-9
                || record.distance.abs() > crate::MAX_COORD
                || !record.sources.iter().all(|(id, _)| id_ok(id) && (!trimmed.contains(id.as_str()) || rebuilt.contains(id.as_str())))
            {
                return Err(invalid());
            }
            if record.sources.iter().any(|(id, _)| self.curve_index(id).is_none()) {
                return Err(bad("an offset source vanished or changed shape"));
            }
            for id in &record.outputs {
                let curve = self.curve_index(id).and_then(|i| self.curves.get(i)).ok_or_else(|| bad("invalid trimmed offset output"))?;
                let (ends, inner): (Vec<usize>, Vec<usize>) = match &curve.kind {
                    CurveKind::Spline { pts, control: false, degree: 3, periodic: false, handles, .. } if handles.is_empty() && pts.len() >= 2 => {
                        let ends = [pts.first(), pts.last()].into_iter().flatten().copied().collect();
                        (ends, pts.iter().skip(1).take(pts.len() - 2).copied().collect())
                    }
                    CurveKind::Arc { c, a, b } => (vec![*a, *b], vec![*c]),
                    _ => return Err(bad("invalid trimmed offset output")),
                };
                work = work.saturating_add(ends.len() + inner.len());
                if !id_ok(id) || record_of.insert(id.as_str(), index).is_some() || work > 1_000_000 {
                    return Err(bad("invalid trimmed offset output"));
                }
                for (point, shared) in ends.iter().map(|p| (*p, true)).chain(inner.iter().map(|p| (*p, false))) {
                    let free = self.points.get(point).is_some_and(|p| p.link.is_none());
                    let (Some(owner), Some(uses)) = (owner.get_mut(point), uses.get_mut(point)) else {
                        return Err(bad("invalid trimmed offset output"));
                    };
                    *uses += 1;
                    if point == 0 || !free || *owner.get_or_insert(index) != index || (!shared && *uses > 1) {
                        return Err(bad("invalid trimmed offset output point roles"));
                    }
                }
            }
            rebuilt.extend(record.outputs.iter().map(String::as_str));
        }
        // No other curve uses a point a trimmed offset places.
        for curve in self.curves.iter().filter(|c| !record_of.contains_key(c.id.as_str())) {
            let points = curve.kind.point_ids();
            work = work.saturating_add(points.len());
            if work > 2_000_000 {
                return Err(bad("offset point ownership work limit exceeded"));
            }
            if points.iter().any(|p| owner.get(*p).is_some_and(Option::is_some)) {
                return Err(bad("a curve uses a point that a trimmed offset places; break the offset first"));
            }
        }
        Ok(())
    }

    /// Rebuild every trimmed offset from its sources at its distance. The curves it made
    /// keep their ids and their points, and take the new shape; a spline that needs more nodes
    /// gets new points at the end of the point table, so no point index changes. Where the
    /// outline no longer has the same curves in the same order (a corner trimmed away, a new
    /// crossing), there is nothing for the old curves to become and the rebuild is refused.
    pub(crate) fn refresh_trimmed_offsets(&mut self) -> Result<()> {
        if !self.offsets.iter().any(|o| o.trimmed) {
            return Ok(());
        }
        self.validate_trimmed_offsets()?;
        let mut candidate = self.clone();
        for record in self.offsets.iter().filter(|o| o.trimmed) {
            candidate.refresh_trimmed_offset(record)?;
        }
        *self = candidate;
        Ok(())
    }

    fn refresh_trimmed_offset(&mut self, record: &crate::OffsetChain) -> Result<()> {
        let changed =
            || bad("the offset's outline changes shape at this distance (a corner or a crossing comes or goes); break the offset or make it again");
        let edited = || bad("offset output was edited; break its association first");
        if record.sources.is_empty()
            || record.sources.len() > 10_000
            || record.outputs.is_empty()
            || record.outputs.len() > 10_000
            || record.topology.is_some()
            || !record.distance.is_finite()
            || record.distance.abs() < 1e-9
            || record.sources.iter().any(|(source, _)| record.outputs.contains(source))
        {
            return Err(bad("damaged offset chain"));
        }
        let order = record
            .sources
            .iter()
            .map(|(id, forward)| self.curve_index(id).map(|i| (i, *forward)).ok_or_else(|| bad("an offset source vanished or changed shape")))
            .collect::<Result<Vec<_>>>()?;
        let old = record.outputs.iter().map(|id| self.curve_index(id).ok_or_else(|| bad("offset output vanished"))).collect::<Result<Vec<_>>>()?;
        // What each output is now: its points in walk order and, for a spline, its nodes.
        let floors = old
            .iter()
            .map(|i| match self.curves.get(*i).map(|c| &c.kind) {
                Some(CurveKind::Spline { pts, control: false, degree: 3, periodic: false, handles, .. }) if handles.is_empty() && pts.len() >= 2 => {
                    Ok(pts.len())
                }
                Some(CurveKind::Arc { .. }) => Ok(0),
                _ => Err(edited()),
            })
            .collect::<Result<Vec<_>>>()?;
        let raw = Raw::chain(self, &order, record.distance)?;
        if raw.closed != record.closed {
            return Err(changed());
        }
        let us = samples(&raw);
        let fitted = match trimmed(&raw, &us, false, &floors) {
            Ok((fitted, _)) => fitted,
            Err(Trim::NothingToTrim) => return Err(changed()),
            Err(Trim::Failed(error)) => return Err(explain(&raw, &us, record.distance, error)),
        };
        if fitted.iter().map(|arc| arc.outs.len()).sum::<usize>() != old.len() {
            return Err(changed());
        }
        // The nodes the new curves share (a crossing, a seam, the joint between two curves of
        // one kept arc) must be the points the old curves share, and no others.
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        enum Node {
            Crossing(usize),
            Start(usize),
            End(usize),
            Inner(usize, usize),
        }
        let mut node_point = std::collections::BTreeMap::new();
        let mut point_node = std::collections::BTreeMap::new();
        let mut pair = |node: Node, point: usize| -> Result<()> {
            if *node_point.entry(node).or_insert(point) != point || *point_node.entry(point).or_insert(node) != node {
                return Err(changed());
            }
            Ok(())
        };
        let mut placed = 0;
        let mut plan = Vec::with_capacity(old.len());
        for (a, arc) in fitted.iter().enumerate() {
            let end_node = |end: End, own: Node| match end {
                End::Crossing(i) => Node::Crossing(i),
                End::Seam => Node::Start(a),
                End::Free => own,
            };
            let count = arc.outs.len();
            for (k, out) in arc.outs.iter().enumerate() {
                let from = if k == 0 { end_node(arc.start, Node::Start(a)) } else { Node::Inner(a, k) };
                let to = if k + 1 < count { Node::Inner(a, k + 1) } else { end_node(arc.end, Node::End(a)) };
                let curve = *old.get(placed).ok_or_else(changed)?;
                placed += 1;
                let (start, end) = match (self.curves.get(curve).map(|c| &c.kind), out) {
                    (Some(CurveKind::Spline { pts, .. }), Out::Spline { .. }) => match (pts.first(), pts.last()) {
                        (Some(first), Some(last)) => (*first, *last),
                        _ => return Err(edited()),
                    },
                    // Sketch arcs run counter-clockwise from `a` to `b`.
                    (Some(CurveKind::Arc { a, b, .. }), Out::Arc { sweep, .. }) => {
                        if *sweep >= 0.0 {
                            (*a, *b)
                        } else {
                            (*b, *a)
                        }
                    }
                    _ => return Err(changed()),
                };
                pair(from, start)?;
                pair(to, end)?;
                plan.push((curve, out, start, end));
            }
        }
        let place = |sk: &mut Sketch, index: usize, at: Vec2| -> Result<()> {
            let point = sk.points.get_mut(index).ok_or_else(|| bad("offset output point vanished"))?;
            if point.link.is_some() {
                return Err(edited());
            }
            point.pos = at;
            point.fixed = true;
            Ok(())
        };
        for (curve, out, start, end) in plan {
            let (start_at, end_at) = out.ends();
            place(self, start, start_at)?;
            place(self, end, end_at)?;
            let kind = match (self.curves.get(curve).map(|c| c.kind.clone()), out) {
                (Some(CurveKind::Spline { pts, .. }), Out::Spline { nodes, .. }) => {
                    let inner = nodes.len().saturating_sub(2);
                    let mut indices = vec![start];
                    for (k, at) in nodes.iter().skip(1).take(inner).enumerate() {
                        let index = match pts.get(k + 1).filter(|_| k + 2 < pts.len()) {
                            Some(index) => *index,
                            None => self.add_point(*at, None)?,
                        };
                        place(self, index, *at)?;
                        indices.push(index);
                    }
                    indices.push(end);
                    // A kept arc that leaves a crossing and comes back to it starts and ends on one point.
                    let distinct = indices.iter().skip(usize::from(start == end)).copied().collect::<std::collections::BTreeSet<_>>().len();
                    if indices.len() < pts.len() || distinct + usize::from(start == end) != indices.len() {
                        return Err(edited());
                    }
                    CurveKind::Spline { pts: indices, control: false, degree: 3, periodic: false, nurbs: None, handles: Vec::new() }
                }
                (Some(CurveKind::Arc { c, .. }), Out::Arc { center, sweep, .. }) => {
                    if c == start || c == end {
                        return Err(edited());
                    }
                    place(self, c, *center)?;
                    if *sweep >= 0.0 { CurveKind::Arc { c, a: start, b: end } } else { CurveKind::Arc { c, a: end, b: start } }
                }
                _ => return Err(changed()),
            };
            self.curves.get_mut(curve).ok_or_else(|| bad("offset output vanished"))?.kind = kind;
        }
        Ok(())
    }

    fn add_trimmed_offset(&mut self, raw: &Raw, us: &[f64], require_crossing: bool) -> std::result::Result<Vec<usize>, Trim> {
        let (fitted, crossings) = trimmed(raw, us, require_crossing, &[])?;
        let mut candidate = self.clone();
        let mut corners: Vec<Option<usize>> = vec![None; crossings];
        let mut outputs = Vec::new();
        for arc in &fitted {
            let mut shared = |end: End, at: Vec2, candidate: &mut Sketch| -> Result<usize> {
                match end {
                    End::Crossing(i) => match corners.get(i).copied().flatten() {
                        Some(index) => Ok(index),
                        None => {
                            let index = candidate.add_point(at, None)?;
                            if let Some(slot) = corners.get_mut(i) {
                                *slot = Some(index);
                            }
                            Ok(index)
                        }
                    },
                    End::Free | End::Seam => candidate.add_point(at, None),
                }
            };
            let (first_at, _) = arc.outs.first().map(Out::ends).ok_or_else(|| bad("empty offset arc"))?;
            let arc_start = shared(arc.start, first_at, &mut candidate)?;
            let mut node = arc_start;
            let count = arc.outs.len();
            for (k, out) in arc.outs.iter().enumerate() {
                let (_, end_at) = out.ends();
                let end = if k + 1 < count {
                    candidate.add_point(end_at, None)?
                } else if arc.end == End::Seam {
                    arc_start
                } else {
                    shared(arc.end, end_at, &mut candidate)?
                };
                let kind = match out {
                    Out::Spline { nodes, .. } => {
                        let mut pts = vec![node];
                        for p in nodes.iter().skip(1).take(nodes.len().saturating_sub(2)) {
                            pts.push(candidate.add_point(*p, None)?);
                        }
                        pts.push(end);
                        CurveKind::Spline { pts, control: false, degree: 3, periodic: false, nurbs: None, handles: Vec::new() }
                    }
                    Out::Arc { center, sweep, .. } => {
                        let c = candidate.add_point(*center, None)?;
                        // Sketch arcs run counter-clockwise from `a` to `b`.
                        if *sweep >= 0.0 { CurveKind::Arc { c, a: node, b: end } } else { CurveKind::Arc { c, a: end, b: node } }
                    }
                };
                outputs.push(candidate.add_curve(kind, None)?);
                node = end;
            }
        }
        // The result is derived geometry, fixed like a linked offset's nodes: a drag must not
        // bend it away from the certified distance. Its points are its own.
        let made = candidate.points.len();
        for point in candidate.points.iter_mut().skip(self.points.len()).take(made) {
            point.fixed = true;
        }
        *self = candidate;
        Ok(outputs)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn spline(coords: &[(f64, f64)], periodic: bool) -> (Sketch, usize) {
        let mut sk = Sketch::new();
        let pts = coords.iter().map(|(x, y)| sk.add_point(Vec2::new(*x, *y), None).unwrap()).collect();
        let source = sk.add_curve(CurveKind::Spline { pts, control: false, degree: 3, periodic, nurbs: None, handles: Vec::new() }, None).unwrap();
        (sk, source)
    }

    /// Distance from `p` to the source by dense sampling (independent of the code under test).
    fn source_distance(segs: &[Seg2], p: Vec2) -> f64 {
        segs.iter().flat_map(|s| (0..=4000).map(move |j| s.point_at(j as f64 / 4000.0).dist(p))).fold(f64::INFINITY, f64::min)
    }

    /// Every output curve stays the offset distance from the source, and (when `closed`)
    /// every output end meets another output's end.
    fn check_offset(sk: &Sketch, source: &[Seg2], curves: &[usize], distance: f64, closed: bool) {
        for c in curves {
            for seg in sk.exact_pieces(*c) {
                for j in 0..=16 {
                    let p = seg.point_at(j as f64 / 16.0);
                    let found = source_distance(source, p);
                    assert!((found - distance).abs() < OFFSET_TOLERANCE, "distance {found} at {p:?}");
                }
            }
        }
        if closed {
            let ends: Vec<Vec2> = curves
                .iter()
                .flat_map(|c| {
                    let segs = sk.exact_pieces(*c);
                    [segs.first().unwrap().start(), segs.last().unwrap().end()]
                })
                .collect();
            for end in &ends {
                assert!(ends.iter().filter(|e| e.dist(*end) < 1e-9).count() >= 2, "an output end at {end:?} joins nothing");
            }
        }
    }

    /// The cookie-cutter outline from the report: a closed spline whose notch near
    /// (4.2, 36) bends with a radius of about 0.3 mm, offset 1.26 mm outwards.
    const COOKIE: [(f64, f64); 16] = [
        (0.0, 0.0),
        (-1.6, 12.8),
        (-7.4, 23.4),
        (-4.8, 26.6),
        (-8.6, 39.0),
        (2.8, 41.6),
        (4.2, 36.0),
        (11.8, 43.2),
        (17.2, 43.2),
        (37.2, 24.6),
        (38.6, 23.0),
        (35.8, 16.4),
        (35.8, 0.0),
        (24.2, 0.0),
        (9.6, 0.0),
        (0.0, 0.0),
    ];

    #[test]
    fn trimmed_offset_of_closed_spline_with_tight_notch() {
        let (mut sk, source) = spline(&COOKIE, true);
        let before = sk.clone();
        let FreeformOffsetOutput::Trimmed(curves) = sk.offset_freeform(source, true, 1.26).unwrap() else {
            panic!("a 1.26 mm offset over a 0.3 mm notch must be trimmed");
        };
        assert!(sk.freeform_offsets.is_empty(), "a trimmed offset is not linked");
        assert_eq!(&sk.points[..before.points.len()], &before.points);
        let exact = sk.exact_segs(source).unwrap();
        check_offset(&sk, &exact, &curves, 1.26, true);
        // The corner left where the loop was cut lies outside the notch, the full
        // distance from the source.
        let CurveKind::Spline { pts, .. } = &sk.curves[curves[0]].kind else { panic!("spline output") };
        assert!((source_distance(&exact, sk.points[pts[0]].pos) - 1.26).abs() < 1e-6);
    }

    #[test]
    fn untrimmed_offsets_stay_linked_and_inward_is_unchanged() {
        let (mut sk, source) = spline(&COOKIE, true);
        assert!(matches!(sk.offset_freeform(source, true, -1.26).unwrap(), FreeformOffsetOutput::Linked(_)));
        assert_eq!(sk.freeform_offsets.len(), 1);
    }

    #[test]
    fn trimmed_offset_of_open_spline_hairpin() {
        let (mut sk, source) = spline(&[(-25.0, -26.0), (2.6, 3.0), (10.9, 10.7), (-13.9, -3.8)], false);
        let FreeformOffsetOutput::Trimmed(curves) = sk.offset_freeform(source, true, 0.6).unwrap() else {
            panic!("the hairpin offset must be trimmed");
        };
        let exact = sk.exact_segs(source).unwrap();
        check_offset(&sk, &exact, &curves, 0.6, false);
    }

    /// A closed chain of cubic Bézier curves, the way SVG outlines import.
    fn chain_of(beziers: &[[(f64, f64); 4]]) -> (Sketch, Vec<(usize, bool)>) {
        let mut sk = Sketch::new();
        let first = sk.add_point(Vec2::new(beziers[0][0].0, beziers[0][0].1), None).unwrap();
        let mut start = first;
        let mut order = Vec::new();
        for (k, b) in beziers.iter().enumerate() {
            let mid = [1, 2].map(|i| sk.add_point(Vec2::new(b[i].0, b[i].1), None).unwrap());
            let end = if k + 1 == beziers.len() { first } else { sk.add_point(Vec2::new(b[3].0, b[3].1), None).unwrap() };
            let c = sk
                .add_curve(
                    CurveKind::Spline {
                        pts: vec![start, mid[0], mid[1], end],
                        control: true,
                        degree: 3,
                        periodic: false,
                        nurbs: None,
                        handles: Vec::new(),
                    },
                    None,
                )
                .unwrap();
            order.push((c, true));
            start = end;
        }
        (sk, order)
    }

    /// An outline like an imported SVG: real corners both ways, a joint with a tiny kink,
    /// and a zero-length handle (pointing roughly along its curve, as drawing programs make
    /// them).
    fn bezier_chain() -> (Sketch, Vec<(usize, bool)>) {
        chain_of(&[
            [(0.0, 0.0), (10.0, -5.0), (20.0, -5.0), (30.0, 0.0)],
            // Corner at (30, 0).
            [(30.0, 0.0), (35.0, 10.0), (35.0, 20.0), (30.0, 30.0)],
            // A tiny kink at (30, 30), a few thousandths of a degree off the incoming tangent.
            [(30.0, 30.0), (25.0 + 0.0002, 40.0), (10.0, 40.0), (5.0, 35.0)],
            // A zero-length handle at the end of this piece.
            [(5.0, 35.0), (4.0, 31.0), (8.0, 20.0), (8.0, 20.0)],
            // Corner at (8, 20), then back to the start (a corner there too).
            [(8.0, 20.0), (0.0, 15.0), (-5.0, 5.0), (0.0, 0.0)],
        ])
    }

    #[test]
    fn chain_offset_rounds_outer_corners_and_trims_inner_ones() {
        for distance in [1.26, -1.26] {
            let (mut sk, order) = bezier_chain();
            let before = sk.clone();
            let source: Vec<Seg2> = order.iter().flat_map(|(c, _)| sk.exact_segs(*c).unwrap()).collect();
            let curves = sk.offset_freeform_chain(&order, distance).unwrap();
            assert_eq!(&sk.points[..before.points.len()], &before.points);
            check_offset(&sk, &source, &curves, distance.abs(), true);
            let arcs = curves.iter().filter(|c| matches!(sk.curves[**c].kind, CurveKind::Arc { .. })).count();
            assert!(arcs >= 1, "an outer corner is rounded on each side ({distance})");
        }
    }

    /// A chain's trimmed offset is linked (#12294): at another distance its curves, corner arcs
    /// included, take the new outline's shape with their ids, and the points that were there
    /// stay where they are in the point table.
    #[test]
    fn chain_offset_follows_its_distance_and_keeps_its_curves() {
        for sign in [1.0, -1.0] {
            let (mut sk, order) = bezier_chain();
            let source: Vec<Seg2> = order.iter().flat_map(|(c, _)| sk.exact_segs(*c).unwrap()).collect();
            let curves = sk.offset_freeform_chain(&order, 1.26 * sign).unwrap();
            let ids: Vec<String> = curves.iter().map(|c| sk.curves[*c].id.clone()).collect();
            assert_eq!(sk.offsets.len(), 1);
            assert!(sk.offsets[0].trimmed && sk.offsets[0].outputs == ids);
            sk.validate_freeform_offsets().unwrap();
            for distance in [1.4, 1.1, 1.26] {
                let before = sk.clone();
                sk.offsets[0].distance = distance * sign;
                sk.refresh_offsets().unwrap();
                assert_eq!(sk.curves.len(), before.curves.len());
                assert!(sk.points.len() >= before.points.len());
                assert!(sk.points.iter().zip(&before.points).all(|(p, q)| p.id == q.id));
                let now: Vec<usize> = ids.iter().map(|id| sk.curve_index(id).unwrap()).collect();
                check_offset(&sk, &source, &now, distance, true);
            }
        }
    }

    #[test]
    fn hostile_trimmed_offset_records_are_refused_not_followed() {
        let (mut sk, order) = bezier_chain();
        sk.offset_freeform_chain(&order, 1.26).unwrap();
        let good = sk.clone();
        let refused = |sk: &Sketch| {
            let mut refreshed = sk.clone();
            let failed = refreshed.refresh_offsets().is_err();
            failed && sk.validate_freeform_offsets().is_err()
        };
        // An output that is its own source, a missing curve, an output of another kind, a
        // distance that is no number, and the same output twice.
        let source = good.offsets[0].sources[0].0.clone();
        let mut sk = good.clone();
        sk.offsets[0].outputs[0] = source.clone();
        assert!(refused(&sk));
        let mut sk = good.clone();
        sk.offsets[0].outputs.push("nothing".into());
        assert!(refused(&sk));
        let mut sk = good.clone();
        sk.offsets[0].sources.push(("nothing".into(), true));
        assert!(refused(&sk));
        let mut sk = good.clone();
        sk.offsets[0].distance = f64::NAN;
        assert!(refused(&sk));
        let mut sk = good.clone();
        let first = sk.offsets[0].outputs[0].clone();
        sk.offsets[0].outputs.push(first);
        assert!(refused(&sk));
        // Points an offset must not move: the origin as a corner arc's centre, a point of a
        // source curve as a spline's node, and one node in two of its splines.
        let outputs: Vec<usize> = good.offsets[0].outputs.iter().map(|id| good.curve_index(id).unwrap()).collect();
        let arc = *outputs.iter().find(|c| matches!(good.curves[**c].kind, CurveKind::Arc { .. })).unwrap();
        let splines: Vec<usize> = outputs.iter().copied().filter(|c| matches!(good.curves[*c].kind, CurveKind::Spline { .. })).collect();
        let mut sk = good.clone();
        if let CurveKind::Arc { c, .. } = &mut sk.curves[arc].kind {
            *c = 0;
        }
        assert!(refused(&sk));
        let taken = good.curves[good.curve_index(&source).unwrap()].kind.point_ids()[0];
        let mut sk = good.clone();
        if let CurveKind::Spline { pts, .. } = &mut sk.curves[splines[0]].kind {
            pts[1] = taken;
        }
        assert!(refused(&sk));
        let mut sk = good.clone();
        let shared = good.curves[splines[0]].kind.point_ids()[1];
        if let CurveKind::Spline { pts, .. } = &mut sk.curves[splines[1]].kind {
            pts[1] = shared;
        }
        assert!(refused(&sk));
        // Another kind of offset that follows a trimmed one, and a trimmed one that follows
        // itself through a second record.
        let mut sk = good.clone();
        sk.offsets.push(crate::OffsetChain {
            sources: vec![(good.offsets[0].outputs[0].clone(), true)],
            outputs: vec![source.clone()],
            distance: 1.0,
            closed: false,
            topology: None,
            trimmed: false,
        });
        assert!(refused(&sk));
        let mut sk = good.clone();
        let mut second = good.offsets[0].clone();
        second.sources = vec![(good.offsets[0].outputs[0].clone(), true)];
        second.outputs = vec![good.offsets[0].sources[0].0.clone()];
        sk.offsets.insert(0, second);
        assert!(refused(&sk));
        // One output fewer than the outline has curves: the outline no longer matches.
        let mut sk = good.clone();
        sk.offsets[0].outputs.pop();
        assert!(sk.refresh_offsets().is_err());
        // An output whose points were rewired elsewhere.
        let mut sk = good.clone();
        let output = sk.curve_index(&good.offsets[0].outputs[0]).unwrap();
        sk.curves[output].kind.map_points(&|_| 0);
        assert!(sk.refresh_offsets().is_err());
    }

    #[test]
    fn sharply_curling_zero_handle_at_a_corner() {
        // A zero-length handle far off its curve's line curls the curve's end sharply
        // (infinite curvature at the end), right before a corner, on both sides.
        let beziers = [
            [(0.0, 0.0), (10.0, -5.0), (20.0, -5.0), (30.0, 0.0)],
            [(30.0, 0.0), (35.0, 10.0), (35.0, 20.0), (30.0, 30.0)],
            [(30.0, 30.0), (25.0, 40.0), (10.0, 40.0), (5.0, 35.0)],
            [(5.0, 35.0), (2.0, 32.0), (8.0, 20.0), (8.0, 20.0)],
            [(8.0, 20.0), (0.0, 15.0), (-5.0, 5.0), (0.0, 0.0)],
        ];
        for distance in [0.5, -0.5, 1.26, -1.26, 3.0, -3.0] {
            let (mut sk, order) = chain_of(&beziers);
            let source: Vec<Seg2> = order.iter().flat_map(|(c, _)| sk.exact_segs(*c).unwrap()).collect();
            let curves = sk.offset_freeform_chain(&order, distance).unwrap();
            check_offset(&sk, &source, &curves, distance.abs(), true);
        }
    }

    #[test]
    fn untrimmable_fold_reports_the_bend() {
        // A single hairpin piece whose offset folds near its open end without crossing back.
        let (mut sk, source) = spline(&[(0.0, 0.0), (10.0, 0.0), (10.2, 0.4)], false);
        let before = sk.clone();
        if let Err(error) = sk.offset_freeform(source, true, 5.0) {
            assert!(error.to_string().contains("bend"), "{error}");
            assert_eq!(sk, before, "a refused offset leaves the sketch unchanged");
        }
    }

    #[test]
    fn hostile_offsets_never_panic() {
        let (sk, source) = spline(&COOKIE, true);
        let (chain, order) = bezier_chain();
        let reversed: Vec<(usize, bool)> = order.iter().rev().map(|(c, f)| (*c, !f)).collect();
        for d in [f64::NAN, f64::INFINITY, 0.0, 1e-12, 1e6, -1e6, 50.0, -50.0, 19.0, 1.26, -1.8, 1.8] {
            for forward in [true, false] {
                let _ = sk.clone().offset_freeform(source, forward, d);
                let _ = chain.clone().offset_freeform_chain(if forward { &order } else { &reversed }, d);
            }
        }
        let _ = chain.clone().offset_freeform_chain(&[], 1.0);
    }
}

//! A sketch offset that has to be trimmed (a chain with a free-form curve in it, such as a
//! spline with a line, or one spline whose offset folds over a tight bend) is saved as an
//! offset association: an entry of the sketch's `offsets` with `"trimmed": true`, whose
//! `outputs` are the curves of the trimmed parallel outline in walk order rather than one per
//! source, driven by an offset dimension like any other offset. An older design has no such
//! entry: its offsets of this kind stay the ordinary curves they were saved as, and nothing
//! drives them. The last release opens a design that has one with a warning about the newer
//! format number, keeps the entry and the dimension as it found them, and cannot rebuild
//! that offset.

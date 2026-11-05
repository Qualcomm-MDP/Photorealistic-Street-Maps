import React from 'react';
import styles from './styles.module.css';

// Illustrative patch sizes (percent of the width, relative heights). In the
// real atlas they follow each face's size in metres, up to 1024 px.
const ROWS = [
  {label: 'row 0', patches: [
    {name: 'patch_0', w: 30, h: 1},
    {name: 'patch_1', w: 20, h: 0.7},
    {name: 'patch_2', w: 26, h: 0.85},
  ]},
  {label: 'row 1', patches: [
    {name: 'patch_3', w: 36, h: 0.75},
    {name: 'patch_4', w: 24, h: 0.6},
  ]},
];

/**
 * The texture atlas layout (src/texturing/tex_projection.py): patches packed
 * left to right in rows, each row as tall as its tallest patch, and a 16 px
 * strip of grey at the bottom for faces no photo covers.
 */
export default function AtlasDiagram() {
  return (
    <figure className={styles.figure}>
      <div className={styles.width} aria-hidden="true">
        <span>4096 px</span>
      </div>
      <div
        className={styles.atlas}
        role="img"
        aria-label="Texture atlas: patches packed left to right in rows, each row as tall as its tallest patch, then a 16 pixel grey strip at the bottom.">
        {ROWS.map((row) => (
          <div key={row.label} className={styles.row}>
            {row.patches.map((p) => (
              <div
                key={p.name}
                className={styles.patch}
                style={{width: `${p.w}%`, height: `${p.h * 100}%`}}>
                {p.name}
              </div>
            ))}
            <span className={styles.rowLabel}>{row.label}</span>
          </div>
        ))}
        <div className={styles.more}>more rows …</div>
        <div className={styles.grey}>untextured grey (225) · 16 px</div>
      </div>
      <figcaption className={styles.caption}>
        Patches are packed left to right; each row is as tall as its tallest
        patch, and a new row starts when the next patch doesn&apos;t fit. Faces
        no photo covers use the grey strip at the bottom.
      </figcaption>
    </figure>
  );
}

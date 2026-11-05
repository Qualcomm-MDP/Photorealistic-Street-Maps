import React from 'react';
import styles from './styles.module.css';

/**
 * A folder/file tree, e.g. the project layout. Folders are open by default
 * and can be collapsed.
 *
 *   <FileTree items={[
 *     {name: 'src', note: 'Application code', children: [
 *       {name: 'main.py', note: 'Command-line entry point'},
 *     ]},
 *   ]} />
 *
 * An item with `children` is a folder; `folder: true` marks an empty one.
 */
export default function FileTree({items}) {
  return (
    <div className={styles.tree}>
      <Items items={items} depth={0} />
    </div>
  );
}

// depth is passed down so each level's name column can shrink by its
// indentation, keeping all the descriptions in one column.
function Items({items, depth}) {
  return (
    <ul className={styles.list} style={{'--depth': depth}}>
      {items.map((item) => (
        <Item key={item.name} item={item} depth={depth} />
      ))}
    </ul>
  );
}

function Item({item, depth}) {
  const isFolder = Boolean(item.children || item.folder);
  const label = (
    <span className={styles.row}>
      <span className={styles.entry}>
        {isFolder ? <FolderIcon /> : <FileIcon />}
        <span className={styles.name}>
          {item.name}
          {isFolder ? '/' : ''}
        </span>
      </span>
      {item.note && <span className={styles.note}>{item.note}</span>}
    </span>
  );
  if (!item.children) return <li className={styles.item}>{label}</li>;
  return (
    <li className={styles.item}>
      <details open className={styles.folder}>
        <summary className={styles.summary}>
          <Chevron />
          {label}
        </summary>
        <Items items={item.children} depth={depth + 1} />
      </details>
    </li>
  );
}

const iconProps = {
  viewBox: '0 0 16 16',
  width: 16,
  height: 16,
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.4,
  strokeLinejoin: 'round',
  'aria-hidden': true,
};

function FolderIcon() {
  return (
    <svg {...iconProps} className={styles.icon}>
      <path d="M1.75 4.25a1 1 0 0 1 1-1h3.1l1.4 1.5h6a1 1 0 0 1 1 1v6.5a1 1 0 0 1-1 1H2.75a1 1 0 0 1-1-1Z" />
    </svg>
  );
}

function FileIcon() {
  return (
    <svg {...iconProps} className={styles.icon}>
      <path d="M3.75 1.75h5.5l3 3v9.5h-8.5Z" />
      <path d="M9.25 1.75v3h3" />
    </svg>
  );
}

function Chevron() {
  return (
    <svg {...iconProps} width={12} height={12} className={styles.chevron}>
      <path d="m5.5 3.5 5 4.5-5 4.5" strokeLinecap="round" />
    </svg>
  );
}

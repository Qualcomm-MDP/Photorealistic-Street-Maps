// @ts-check
// Docusaurus config for the PSM documentation site.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

const repo = 'https://github.com/Qualcomm-MDP/PSM';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'PSM',
  tagline: 'Photorealistic 3D street maps from open data',
  favicon: 'img/favicon.svg',

  future: {
    v4: true,
  },

  // GitHub Pages: https://qualcomm-mdp.github.io/PSM/
  url: 'https://qualcomm-mdp.github.io',
  baseUrl: '/PSM/',
  organizationName: 'Qualcomm-MDP',
  projectName: 'PSM',
  trailingSlash: false,

  onBrokenLinks: 'throw',

  markdown: {
    // .md files are plain Markdown (CommonMark), .mdx files are MDX. The API
    // pages are generated from docstrings, which can contain characters
    // (<, {) that MDX would try to parse.
    format: 'detect',
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },
  themes: ['@docusaurus/theme-mermaid'],

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          // The docs are the whole site, served from its root.
          routeBasePath: '/',
          sidebarPath: './sidebars.js',
          editUrl: `${repo}/tree/main/website/`,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'PSM',
        items: [
          {type: 'docSidebar', sidebarId: 'docs', position: 'left', label: 'Docs'},
          {type: 'docSidebar', sidebarId: 'api', position: 'left', label: 'API'},
          {type: 'docSidebar', sidebarId: 'contributing', position: 'left', label: 'Contributing'},
          {href: repo, label: 'GitHub', position: 'right'},
        ],
      },
      footer: {
        style: 'light',
        links: [
          {
            title: 'Docs',
            items: [
              {label: 'Getting Started', to: '/getting-started'},
              {label: 'Web App', to: '/web-app'},
              {label: 'Pipeline', to: '/pipeline/overview'},
            ],
          },
          {
            title: 'Project',
            items: [
              {label: 'GitHub', href: repo},
              {label: 'Contributing', to: '/contributing'},
            ],
          },
        ],
        copyright:
          'Built at the University of Michigan in partnership with Qualcomm. Map data © OpenStreetMap contributors.',
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['bash', 'python', 'json'],
      },
    }),
};

export default config;

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docs: [
    'index',
    'getting-started',
    'web-app',
    'configuration',
    'architecture',
    {
      type: 'category',
      label: 'Pipeline',
      collapsed: false,
      items: [
        'pipeline/overview',
        'pipeline/data-processing',
        'pipeline/terrain-and-aerial',
        'pipeline/mesh-generation',
        'pipeline/texturing',
        'pipeline/evaluation',
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      items: [
        'api/pipeline',
        'api/providers',
        'api/utilities',
        'api/reporter',
        'api/profiler',
        'api/web',
      ],
    },
    'contributing',
  ],
};

export default sidebars;

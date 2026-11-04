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
    'contributing',
  ],
  // The API reference has its own tab in the navbar and its own sidebar.
  api: [
    'api/index',
    {
      type: 'category',
      label: 'Python',
      collapsed: false,
      items: [
        'api/pipeline',
        'api/providers',
        'api/utilities',
        'api/reporter',
        'api/profiler',
      ],
    },
    {
      type: 'category',
      label: 'Web app',
      collapsed: false,
      items: ['api/web'],
    },
  ],
};

export default sidebars;

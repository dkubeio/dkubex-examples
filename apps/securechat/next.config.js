const { i18n } = require('./next-i18next.config');

/** @type {import('next').NextConfig} */
const nextConfig = {
  i18n,
  reactStrictMode: true,
  trailingSlash: true,

  async rewrites() {
    return [
      {
        source: "/api/chat/:path*",
        destination: "http://127.0.0.1:8877/stream/:path*"
      },
      {
        source: "/api/title/:path*",
        destination: "http://127.0.0.1:8877/title/:path*"
      },
    ];
  },
  env: {
    NEXT_PUBLIC_BASE_URL: process.env.INGRESS_PREFIX,
    BASE_URL: process.env.INGRESS_PREFIX
  },
  basePath: process.env.INGRESS_PREFIX,
  // async redirects() {
  //   return [
  //       {
  //           source: '/',
  //           destination: process.env.BASE_URL ?? '',
  //           basePath: false,
  //           permanent: false
  //       }
  //   ]
  // },
  webpack(config, { isServer, dev }) {
    config.experiments = {
      asyncWebAssembly: true,
      layers: true,
    };

    return config;
  },
};

module.exports = nextConfig;

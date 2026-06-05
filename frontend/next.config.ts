/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow proxying large audio files to the backend
  experimental: {
    serverActions: {
      bodySizeLimit: "52mb",
    },
  },
};

export default nextConfig;

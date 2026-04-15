/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        cyber: {
          cyan: '#00f3ff',
          blue: '#0055ff',
          dark: '#0a0a0f',
          gray: '#1f1f2e',
          red: '#ff003c',
          amber: '#ffb000'
        }
      }
    },
  },
  plugins: [],
}

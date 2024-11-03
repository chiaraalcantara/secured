/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}", // Ensure this includes your TSX files
  ],
  theme: {
    extend: {
      // Setting colors for theme of application
      colors: {
        primary: {
          50: '#eef2ff',
          100: '#d2daff',
          200: '#a9b8ff',
          300: '#8095ff',
          400: '#5f76ff',
          500: '#465FF1',  // Base
          600: '#3d54cc',
          700: '#3446a3',
          800: '#2c387a',
          900: '#232b61',
        },
        secondary: {
          50: '#f9fbff',
          100: '#f0f4ff',
          200: '#e1e9ff',
          300: '#d3deff',
          400: '#c6d3ff',
          500: '#ECF0FF',  // Base
          600: '#d1dbff',
          700: '#99a8ff',
          800: '#7a89ff',
          900: '#606eff',
        },
      },
      fontFamily: {
        inter: ['Inter', 'sans-serif'], // Custom font ([primary font, fallback font])
      },
    },
  },
  plugins: [],
}
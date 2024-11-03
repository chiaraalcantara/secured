/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}", // Ensure this includes your TSX files
  ],
  theme: {
    extend: {
      colors: {
        primary: '#465FF1', // Custom primary color
        secondary: '#ECF0FF', // Custom secondary color
      },
    },
  },
  plugins: [],
}
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const distConfig = path.join(root, "dist", "config");
const sourceExample = path.join(root, "config", "lab.local.example.json");

fs.mkdirSync(distConfig, { recursive: true });
fs.copyFileSync(sourceExample, path.join(distConfig, "lab.local.example.json"));
console.log("Copied public config example to dist/config/lab.local.example.json");

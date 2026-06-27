const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..");
const repoRoot = path.resolve(root, "..");
const required = [
  path.join(root, "config", "lab.local.example.json"),
  path.join(root, "electron", "main.js"),
  path.join(root, "electron", "preload.js"),
  path.join(root, "src", "renderer.js"),
  path.join(root, "src", "index.html"),
  path.join(repoRoot, "lab", "scripts", "generate_reddit_post_template.py"),
  path.join(repoRoot, "lab", "scripts", "check_public_release_safety.py")
];

const missing = required.filter((filePath) => !fs.existsSync(filePath));
if (missing.length) {
  throw new Error(`Missing public release files:\n${missing.join("\n")}`);
}

const example = JSON.parse(fs.readFileSync(required[0], "utf8"));
const placeholderOk =
  example.lab_root === "<LAB_ROOT>" &&
  example.python_exe === "<LAB_ROOT>/.venv/Scripts/python.exe";

if (!placeholderOk) {
  throw new Error("config/lab.local.example.json must use public placeholders.");
}

console.log(JSON.stringify({
  ok: true,
  mode: "public-release-smoke",
  requires_private_data: false,
  config_example: required[0],
  lab_scripts_present: true
}, null, 2));

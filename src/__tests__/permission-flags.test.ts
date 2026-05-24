import { describe, expect, it } from "bun:test";

const { permissionFlags } = await import("../index.js");

describe("gemini permissionFlags", () => {
  it("fullAuto → --yolo", () => {
    expect(permissionFlags("fullAuto")).toEqual(["--yolo"]);
  });

  it("plan → no flag (interactive/default approval)", () => {
    expect(permissionFlags("plan")).toEqual([]);
  });

  it("acceptEdits → no flag", () => {
    expect(permissionFlags("acceptEdits")).toEqual([]);
  });

  it("undefined / unknown → acceptEdits (no --yolo)", () => {
    expect(permissionFlags(undefined)).toEqual([]);
    expect(permissionFlags("bogus" as never)).toEqual([]);
  });
});

import { LibraryProvider } from "./types";

export declare class EmccWASI implements LibraryProvider {
  imports: Record<string, any>;
  start: (inst: WebAssembly.Instance) => void;
}

export default EmccWASI;

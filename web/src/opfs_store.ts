/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

export type OPFSAccessMode = "async" | "sync" | "auto";

type OPFSEffectiveAccessMode = "async" | "sync";
type OPFSSyncAccessHandleMode = "read-only" | "readwrite";

interface OPFSWritableFileStream extends WritableStream<Uint8Array> {
  write(value: Blob | BufferSource | Uint8Array | string): Promise<void>;
  close(): Promise<void>;
}

interface OPFSFileHandle {
  getFile(): Promise<Blob>;
  createWritable(): Promise<OPFSWritableFileStream>;
  createSyncAccessHandle?: (options?: {
    mode?: OPFSSyncAccessHandleMode;
  }) => Promise<OPFSSyncAccessHandle>;
}

interface OPFSDirectoryHandle {
  getDirectoryHandle(
    name: string,
    options?: { create?: boolean },
  ): Promise<OPFSDirectoryHandle>;
  getFileHandle(
    name: string,
    options?: { create?: boolean },
  ): Promise<OPFSFileHandle>;
  removeEntry(name: string): Promise<void>;
}

interface OPFSStorageManager {
  getDirectory?: () => Promise<OPFSDirectoryHandle>;
}

interface OPFSStoreRecord {
  url: string;
  nbytes: number;
  contentType?: string;
}

interface OPFSStoredEntry {
  payloadHandle: OPFSFileHandle;
  record: OPFSStoreRecord;
}

interface OPFSSyncAccessHandle {
  getSize(): number;
  read(buffer: BufferSource, options?: { at?: number }): number;
  write(buffer: BufferSource, options?: { at?: number }): number;
  truncate(size: number): void;
  flush(): void;
  close(): void;
}

type OPFSGlobalScope = typeof globalThis & {
  DedicatedWorkerGlobalScope?: new () => object;
  FileSystemFileHandle?: {
    prototype?: {
      createSyncAccessHandle?: unknown;
    };
  };
};

const HASH_ALGORITHM = "SHA-256";
const OPFS_STORE_ROOT_DIRECTORY = "tvmjs-opfs-store";

export class OPFSStore {
  private readonly scope: string;
  private readonly requestedAccessMode: OPFSAccessMode;
  private accessMode: OPFSEffectiveAccessMode;
  private directoryPromise?: Promise<OPFSDirectoryHandle>;

  constructor(scope: string, accessMode: OPFSAccessMode = "async") {
    this.scope = scope;
    this.requestedAccessMode = accessMode;
    this.accessMode = OPFSStore.resolveAccessMode(accessMode);
  }

  static isAvailable(): boolean {
    const storage = OPFSStore.getStorageManager();
    return storage !== undefined && typeof storage.getDirectory === "function";
  }

  private static resolveAccessMode(
    accessMode: OPFSAccessMode,
  ): OPFSEffectiveAccessMode {
    if (accessMode !== "auto") {
      return accessMode;
    }
    return OPFSStore.isDedicatedWorkerWithSyncAccessHandle()
      ? "sync"
      : "async";
  }

  async has(url: string): Promise<boolean> {
    try {
      const entry = await this.getStoredEntry(url);
      if (entry === undefined) {
        return false;
      }
      return this.hasExpectedPayloadSize(entry);
    } catch (err) {
      if (this.handleCacheMissStateError(err)) {
        return false;
      }
      throw err;
    }
  }

  async read(url: string): Promise<Response | undefined> {
    try {
      const entry = await this.getStoredEntry(url);
      if (entry === undefined) {
        return undefined;
      }
      const blob = await entry.payloadHandle.getFile();
      if (blob.size !== entry.record.nbytes) {
        return undefined;
      }
      return new Response(blob, this.getResponseInit(entry.record));
    } catch (err) {
      if (this.handleCacheMissStateError(err)) {
        return undefined;
      }
      throw err;
    }
  }

  async readArrayBuffer(url: string): Promise<ArrayBuffer | undefined> {
    try {
      const entry = await this.getStoredEntry(url);
      if (entry === undefined) {
        return undefined;
      }
      const payload = await this.readPayload(entry.payloadHandle);
      return payload.byteLength === entry.record.nbytes ? payload : undefined;
    } catch (err) {
      if (this.handleCacheMissStateError(err)) {
        return undefined;
      }
      throw err;
    }
  }

  async write(url: string, response: Response): Promise<void> {
    try {
      const directory = await this.getScopedDirectory();
      const baseName = await this.hashUrl(url);
      await this.removeEntryIfExists(
        directory,
        this.getRecordFilename(baseName),
      );
      const payloadHandle = await directory.getFileHandle(
        this.getPayloadFilename(baseName),
        { create: true },
      );
      const nbytes = await this.writePayload(payloadHandle, response);
      const recordHandle = await directory.getFileHandle(
        this.getRecordFilename(baseName),
        { create: true },
      );
      const record: OPFSStoreRecord = {
        url,
        nbytes,
        contentType: response.headers.get("content-type") ?? undefined,
      };
      await this.writeRecord(recordHandle, record);
    } catch (err) {
      this.resetDirectoryOnInvalidStateError(err);
      throw err;
    }
  }

  async remove(url: string): Promise<void> {
    try {
      const directory = await this.getScopedDirectory();
      const baseName = await this.hashUrl(url);
      await this.removeEntryIfExists(
        directory,
        this.getPayloadFilename(baseName),
      );
      await this.removeEntryIfExists(
        directory,
        this.getRecordFilename(baseName),
      );
    } catch (err) {
      this.resetDirectoryOnInvalidStateError(err);
      throw err;
    }
  }

  private async getStoredEntry(
    url: string,
  ): Promise<OPFSStoredEntry | undefined> {
    const directory = await this.getScopedDirectory();
    const baseName = await this.hashUrl(url);
    const recordHandle = await this.getFileHandleIfExists(
      directory,
      this.getRecordFilename(baseName),
      false,
    );
    if (recordHandle === undefined) {
      return undefined;
    }
    const record = await this.readRecord(recordHandle);
    if (record === undefined || record.url !== url) {
      return undefined;
    }
    const payloadHandle = await this.getFileHandleIfExists(
      directory,
      this.getPayloadFilename(baseName),
      false,
    );
    return payloadHandle === undefined ? undefined : { payloadHandle, record };
  }

  private static getStorageManager(): OPFSStorageManager | undefined {
    if (typeof navigator === "undefined") {
      return undefined;
    }
    return navigator.storage as unknown as OPFSStorageManager;
  }

  private static isDedicatedWorkerWithSyncAccessHandle(): boolean {
    const scope = globalThis as OPFSGlobalScope;
    return (
      typeof scope.DedicatedWorkerGlobalScope === "function" &&
      globalThis instanceof scope.DedicatedWorkerGlobalScope &&
      typeof scope.FileSystemFileHandle?.prototype?.createSyncAccessHandle ===
        "function"
    );
  }

  private async getScopedDirectory(): Promise<OPFSDirectoryHandle> {
    if (this.directoryPromise !== undefined) {
      return this.directoryPromise;
    }
    // Cache scoped directory handle to avoid repeated tree traversal
    this.directoryPromise = (async () => {
      const storage = OPFSStore.getStorageManager();
      if (storage === undefined || typeof storage.getDirectory !== "function") {
        throw new Error("OPFSStore: OPFS API unavailable.");
      }
      let directory = await storage.getDirectory();
      directory = await directory.getDirectoryHandle(OPFS_STORE_ROOT_DIRECTORY, {
        create: true,
      });
      const scopeParts = this.scope.split("/").filter((part) => part.length > 0);
      for (const part of scopeParts) {
        directory = await directory.getDirectoryHandle(
          encodeURIComponent(part),
          { create: true },
        );
      }
      return directory;
    })();
    return this.directoryPromise;
  }

  private async readRecord(
    fileHandle: OPFSFileHandle,
  ): Promise<OPFSStoreRecord | undefined> {
    try {
      const text = await (await fileHandle.getFile()).text();
      const parsed = JSON.parse(text);
      if (
        parsed === undefined ||
        parsed === null ||
        typeof parsed !== "object" ||
        typeof parsed.url !== "string" ||
        !Number.isSafeInteger(parsed.nbytes) ||
        parsed.nbytes < 0
      ) {
        return undefined;
      }
      const record: OPFSStoreRecord = {
        url: parsed.url,
        nbytes: parsed.nbytes,
      };
      if (typeof parsed.contentType === "string") {
        record.contentType = parsed.contentType;
      }
      return record;
    } catch (err) {
      if (
        OPFSStore.getErrorName(err) === "SyntaxError" ||
        this.handleCacheMissStateError(err)
      ) {
        return undefined;
      }
      throw err;
    }
  }

  private getResponseInit(
    record: OPFSStoreRecord,
  ): ResponseInit | undefined {
    return record.contentType !== undefined
      ? { headers: { "content-type": record.contentType } }
      : undefined;
  }

  private async writeRecord(
    handle: OPFSFileHandle,
    record: OPFSStoreRecord,
  ): Promise<void> {
    const writable = await handle.createWritable();
    try {
      await writable.write(new TextEncoder().encode(JSON.stringify(record)));
      await writable.close();
    } catch (err) {
      try {
        await writable.abort();
      } catch {
        // Preserve the original write error.
      }
      throw err;
    }
  }

  private async readPayload(handle: OPFSFileHandle): Promise<ArrayBuffer> {
    const syncHandle = await this.openSyncAccessHandle(handle, "read-only");
    return syncHandle !== undefined
      ? this.readPayloadWithSyncHandle(syncHandle)
      : (await handle.getFile()).arrayBuffer();
  }

  private async hasExpectedPayloadSize(
    entry: OPFSStoredEntry,
  ): Promise<boolean> {
    if (this.accessMode === "sync") {
      const syncHandle = await this.openSyncAccessHandle(
        entry.payloadHandle,
        "read-only",
      );
      if (syncHandle !== undefined) {
        try {
          return syncHandle.getSize() === entry.record.nbytes;
        } finally {
          syncHandle.close();
        }
      }
    }
    const blob = await entry.payloadHandle.getFile();
    return blob.size === entry.record.nbytes;
  }

  private async writePayload(
    handle: OPFSFileHandle,
    response: Response,
  ): Promise<number> {
    const syncHandle = await this.openSyncAccessHandle(handle, "readwrite");
    if (syncHandle !== undefined) {
      return this.writePayloadWithSyncHandle(syncHandle, response);
    }
    return this.writePayloadWithWritable(handle, response);
  }

  private async writePayloadWithWritable(
    handle: OPFSFileHandle,
    response: Response,
  ): Promise<number> {
    const writable = await handle.createWritable();
    try {
      if (response.body !== null) {
        let nbytes = 0;
        const reader = response.body.getReader();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              break;
            }
            await writable.write(value);
            nbytes += value.byteLength;
          }
        } finally {
          reader.releaseLock();
        }
        await writable.close();
        return nbytes;
      }
      const payload = await response.arrayBuffer();
      await writable.write(payload);
      await writable.close();
      return payload.byteLength;
    } catch (err) {
      try {
        await writable.abort();
      } catch {
        // Preserve the original write error.
      }
      throw err;
    }
  }

  private readPayloadWithSyncHandle(
    syncHandle: OPFSSyncAccessHandle,
  ): ArrayBuffer {
    try {
      const size = syncHandle.getSize();
      const payload = new ArrayBuffer(size);
      syncHandle.read(new Uint8Array(payload), { at: 0 });
      return payload;
    } finally {
      syncHandle.close();
    }
  }

  private async writePayloadWithSyncHandle(
    syncHandle: OPFSSyncAccessHandle,
    response: Response,
  ): Promise<number> {
    try {
      syncHandle.truncate(0);
      let offset = 0;
      if (response.body !== null) {
        const reader = response.body.getReader();
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) {
              break;
            }
            syncHandle.write(value, { at: offset });
            offset += value.byteLength;
          }
        } finally {
          reader.releaseLock();
        }
      } else {
        const payload = await response.arrayBuffer();
        syncHandle.write(new Uint8Array(payload), { at: 0 });
        offset = payload.byteLength;
      }
      syncHandle.flush();
      return offset;
    } finally {
      syncHandle.close();
    }
  }

  private async openSyncAccessHandle(
    handle: OPFSFileHandle,
    mode: OPFSSyncAccessHandleMode,
  ): Promise<OPFSSyncAccessHandle | undefined> {
    if (this.accessMode === "async") {
      return undefined;
    }
    if (typeof handle.createSyncAccessHandle !== "function") {
      throw this.createSyncUnavailableError();
    }
    try {
      return await handle.createSyncAccessHandle({ mode });
    } catch (err) {
      const isLockContention =
        OPFSStore.getErrorName(err) === "NoModificationAllowedError";
      if (this.requestedAccessMode === "auto" && isLockContention) {
        return undefined;
      }
      throw err;
    }
  }

  private async getFileHandleIfExists(
    directory: OPFSDirectoryHandle,
    filename: string,
    create: boolean,
  ): Promise<OPFSFileHandle | undefined> {
    try {
      return await directory.getFileHandle(filename, { create });
    } catch (err) {
      if (OPFSStore.isNotFoundError(err)) {
        // NotFound maps to cache miss semantics
        return undefined;
      }
      throw err;
    }
  }

  private async removeEntryIfExists(
    directory: OPFSDirectoryHandle,
    filename: string,
  ): Promise<void> {
    try {
      await directory.removeEntry(filename);
    } catch (err) {
      if (OPFSStore.isNotFoundError(err)) {
        // Delete is intentionally idempotent for missing entries
        return;
      }
      throw err;
    }
  }

  private async hashUrl(url: string): Promise<string> {
    const textEncoder = new TextEncoder();
    const input = textEncoder.encode(url);
    if (
      typeof crypto === "undefined" ||
      crypto.subtle === undefined ||
      typeof crypto.subtle.digest !== "function"
    ) {
      throw new Error("OPFSStore: crypto.subtle.digest is unavailable.");
    }
    const digest = await crypto.subtle.digest(HASH_ALGORITHM, input);
    return Array.from(new Uint8Array(digest))
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");
  }

  private static isNotFoundError(err: unknown): boolean {
    return OPFSStore.getErrorName(err) === "NotFoundError";
  }


  private static isCacheMissStateError(err: unknown): boolean {
    const name = OPFSStore.getErrorName(err);
    return name === "NotFoundError" || name === "InvalidStateError";
  }

  private handleCacheMissStateError(err: unknown): boolean {
    if (!OPFSStore.isCacheMissStateError(err)) {
      return false;
    }
    this.resetDirectoryOnInvalidStateError(err);
    return true;
  }

  private resetDirectoryOnInvalidStateError(err: unknown): void {
    if (OPFSStore.getErrorName(err) === "InvalidStateError") {
      this.directoryPromise = undefined;
    }
  }

  private static getErrorName(err: unknown): string | undefined {
    if (err && typeof err === "object" && "name" in err) {
      const name = (err as { name?: unknown }).name;
      return typeof name === "string" ? name : undefined;
    }
    return undefined;
  }

  private getPayloadFilename(baseName: string): string {
    return `${baseName}.bin`;
  }

  private getRecordFilename(baseName: string): string {
    return `${baseName}.record.json`;
  }

  private createSyncUnavailableError(): Error {
    const err = new Error(
      "OPFSStore: createSyncAccessHandle unavailable; sync OPFS access requires a supported dedicated worker context.",
    );
    err.name = "NotSupportedError";
    return err;
  }
}

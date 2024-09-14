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

export interface NDArrayCacheEntry {
  name: string;
  shape: Array<number>;
  dtype: string;
  format: "f32-to-bf16" | "raw";
  byteOffset: number;
  nbytes: number;
}

export interface NDArrayShardEntry {
  dataPath: string;
  format: "raw-shard";
  nbytes: number;
  records: Array<NDArrayCacheEntry>;
}

/**
 *   Common Interface for the artifact cache
 */
export interface ArtifactCacheTemplate {
  /**
   * Retrieve data object that corresponds to `url` from cache. If data object does not exist in
   * cache, fetch the data and then add to cache.
   *
   * @param url: The url to the data to be cached.
   * @param storetype: This field is required so that `ArtifactIndexedDBCache` can store the
   * actual data object (see `addToCache()`), while `ArtifactCache` which uses the Cache API can
   * return the actual data object rather than the request. There are two options:
   * 1. "json": returns equivalent to `fetch(url).json()`
   * 2. "arraybuffer": returns equivalent to `fetch(url).arraybuffer()`
   * @param signal: An optional AbortSignal allowing user to abort the fetching before its completion.
   * @return The data object (i.e. users do not need to call `.json()` or `.arraybuffer()`).
   *
   * @note This is an async function.
   */
  fetchWithCache(url: string, storetype?: string, signal?: AbortSignal): Promise<any>;

  /**
   * Fetch data from url and add into cache. If already exists in cache, should return instantly.
   *
   * @param url: The url to the data to be cached.
   * @param storetype: Only applies to `ArtifactIndexedDBCache`. Since `indexedDB` stores the actual
   * @param signal: An optional AbortSignal to abort data retrival
   * data rather than a request, we specify `storagetype`. There are two options:
   * 1. "json": IndexedDB stores `fetch(url).json()`
   * 2. "arraybuffer": IndexedDB stores `fetch(url).arrayBuffer()`
   *
   * @note This is an async function.
   */
  addToCache(url: string, storetype?: string, signal?: AbortSignal): Promise<void>;

  /**
   * check if cache has all keys in Cache
   *
   * @note This is an async function.
   */
  hasAllKeys(keys: string[]): Promise<boolean>;

  /**
   * Delete url in cache if url exists
   *
   * @note This is an async function.
   */
  deleteInCache(url: string): Promise<void>;
}


/**
 * Cache to store model related data, implemented with the Cache API.
 */
export class ArtifactCache implements ArtifactCacheTemplate {
  private scope: string;
  private cache?: Cache;

  constructor(scope: string) {
    this.scope = scope;
  }

  /**
   * Convert the Response object to the expected storetype instead
   */
  async responseTostoretype(response: Response, storetype?: string): Promise<any> {
    if (storetype === undefined) {
      return response;
    } else if (storetype.toLowerCase() === "json") {
      return await response.json();
    } else if (storetype.toLowerCase() === "arraybuffer") {
      return await response.arrayBuffer();
    } else {
      console.error("Unknown storage type " + storetype + ", returning raw response");
      return response;
    }
  }

  /**
   * fetch the corresponding url object in response or stored object format
   * @param url url
   * @param storetype the storage type for indexedDB
   * @param signal an optional abort signal to abort fetching
   * @returns response in json, arraybuffer or pure response format
   */
  async fetchWithCache(url: string, storetype?: string, signal?: AbortSignal): Promise<any> {
    await this.addToCache(url, storetype, signal);
    const result = await this.cache.match(new Request(url));
    if (result === undefined) {
      // Already called `addToCache()`, should expect the request in cache.
      throw Error("Cannot fetch " + url);
    }
    return await this.responseTostoretype(result, storetype);
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async addToCache(url: string, storetype?: string, signal?: AbortSignal) {
    const request = new Request(url, signal ? { signal } : undefined);
    if (this.cache === undefined) {
      this.cache = await caches.open(this.scope);
    }
    const result = await this.cache.match(request);
    if (result === undefined) {
      await this.cache.add(request);
    }
  }

  /**
   * Determine if all keys exist in the cache
   * @param keys the url key list of the strings
   * @returns boolean value indicate if all keys are in cache
   */
  async hasAllKeys(keys: string[]) {
    if (this.cache === undefined) {
      this.cache = await caches.open(this.scope);
    }
    return this.cache.keys()
      .then(requests => requests.map(request => request.url))
      .then(cacheKeys => keys.every(key => cacheKeys.indexOf(key) !== -1))
      .catch(() => false);
  }

  /**
   * Delete the corresponding url object in cache
   * @param url the corresponding url object to be deleted
   */
  async deleteInCache(url: string) {
    if (this.cache === undefined) {
      this.cache = await caches.open(this.scope);
    }
    await this.cache.delete(url);
  }
}

/**
 * Cache by IndexedDB to support caching model data
 */
export class ArtifactIndexedDBCache implements ArtifactCacheTemplate {
  private dbName?: string;
  private dbVersion = 1;
  private db: IDBDatabase | undefined;

  constructor(dbName: string) {
    this.dbName = dbName;
  }

  /**
   * Init the indexed DB database if it is not initialized.
   */
  private async initDB() {
    if (this.db != null) {
      return; // the db is already inialized
    }
    return new Promise<void>((resolve, reject) => {
      const request = indexedDB.open(this.dbName, this.dbVersion);
      request.onupgradeneeded = (event) => {
        this.db = (event.target as IDBOpenDBRequest).result;
        if (!this.db.objectStoreNames.contains('urls')) {
          this.db.createObjectStore('urls', { keyPath: 'url' });
        }
      };
      request.onsuccess = (event) => {
        this.db = (event.target as IDBOpenDBRequest).result;
        resolve();
      };
      request.onerror = (event) => {
        console.error("Database error: ", (event.target as IDBOpenDBRequest).error);
        reject((event.target as IDBOpenDBRequest).error);
      };
    });
  }

  /**
   * Check if current url object is in indexedDB or not
   * @param url the url link
   * @returns boolean indicate if url object in indexedDB
   */
  private async isUrlInDB(url: string): Promise<boolean> {
    return new Promise<boolean>((resolve, reject) => {
      const transaction = this.db?.transaction(['urls'], 'readonly');
      if (transaction === undefined) {
        return false;
      }
      const store = transaction.objectStore('urls');
      const request = store.get(url);
      request.onsuccess = () => {
        resolve(request.result !== undefined);
      };
      request.onerror = (event) => {
        reject((event.target as IDBRequest).error);
      };
    });
  }

  async asyncGetHelper(url: string): Promise<any> {
    return new Promise((resolve, reject) => {
      let result: any;
      const transaction = this.db?.transaction(['urls'], 'readonly');
      if (transaction === undefined) {
        return false;
      }
      transaction.oncomplete = () => resolve(result);
      transaction.onerror = () => reject(transaction.error);
      const objectStore = transaction.objectStore('urls');
      const getRequest = objectStore.get(url);
      getRequest.onsuccess = () => {
        result = getRequest.result;
      }
    })
  }

  async fetchWithCache(url: string, storetype?: string, signal?: AbortSignal): Promise<any> {
    await this.addToCache(url, storetype, signal);
    let result = await this.asyncGetHelper(url);
    if (result === null) {
      // previously null data in cache or somehow failed to add to cache, delete and retry
      await this.deleteInCache(url);
      await this.addToCache(url, storetype);
      result = await this.asyncGetHelper(url);
    }
    if (result != null && typeof result === "object" && "data" in result) {
      // `storetype` not used here because the data stored in indexedDB is already in that type
      return result.data;
    }
    throw Error("ArtifactIndexedDBCache failed to fetch: " + url);
  }

  async addToIndexedDB(url: string, response: any, storetype?: string) {
    await this.initDB();
    let data: any;
    // IndexedDB, unlike the Cache API, stores the actual data object, so we convert reponse here.
    if (storetype != undefined) {
      if (storetype.toLowerCase() === "json") {
        data = await response.json();
      } else if (storetype.toLocaleLowerCase() === "arraybuffer") {
        data = await response.arrayBuffer();
      } else {
        throw Error("Unsupported storetyp for IndexedDB: " + storetype);
      }
    }
    return new Promise<void>((resolve, reject) => {
      const transaction = this.db?.transaction(['urls'], 'readwrite');
      if (transaction === undefined) {
        return;
      }
      const store = transaction.objectStore('urls');
      const request = store.add({ data, url }); // Index DB follows a {value, key} format, instead of {key, value} format!
      request.onsuccess = () => resolve();
      request.onerror = (event) => reject((event.target as IDBRequest).error);
    });
  }

  async addToCache(url: string, storetype?: string, signal?: AbortSignal): Promise<void> {
    await this.initDB(); // await the initDB process
    // If already cached, nothing to do
    const isInDB = await this.isUrlInDB(url);
    if (isInDB) {
      return;
    }
    try {
      const response = await fetch(url, signal ? { signal } : undefined);
      if (!response.ok) {
        throw new Error('Network response was not ok');
      }
      const response_copy = response.clone();
      await this.addToIndexedDB(url, response_copy, storetype);
    } catch (error) {
      throw Error("Failed to store " + url + " with error: " + error);
    }
  }

  async hasAllKeys(keys: string[]): Promise<boolean> {
    await this.initDB(); // Ensure the DB is initialized
    if (!this.db) {
      throw new Error('Database is not initialized');
    }
    return new Promise<boolean>((resolve, reject) => {
      const transaction = this.db.transaction(['urls'], 'readonly');
      const store = transaction.objectStore('urls');
      const promises = keys.map(key => {
        return new Promise<boolean>((resolve) => {
          const request = store.get(key);
          request.onsuccess = () => {
            if (request.result === undefined) {
              resolve(false); // Key not found, resolve with false
            } else {
              resolve(true); // Key found, resolve with true
            }
          };
          request.onerror = () => {
            resolve(false); // On error, resolve as if the key was not found
          };
        });
      });
      Promise.all(promises).then(results => {
        const allExist = results.every(exists => exists);
        resolve(allExist);
      }).catch(error => {
        reject(error); // Reject the main promise if any of the promises are rejected
      });
    });
  }

  async deleteInCache(url: string) {
    await this.initDB(); // Make sure the DB is initialized
    const transaction = this.db?.transaction(['urls'], 'readwrite');
    if (transaction === undefined) {
      return;
    }
    const store = transaction.objectStore('urls');
    const request = store.delete(url);
    // Await completion of the delete request
    await new Promise<void>((resolve, reject) => {
      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
    return;
  }
}


/**
 * Function to check if NDarray is in Cache or not
 *
 * @param ndarrayCacheUrl The cache url which links to the NDArray
 * @param cacheScope The scope identifier of the cache
 * @param cacheType The type of the cache: "cache" or "indexedDB"
 * @returns the result if the cache has NDArray
 */
export async function hasNDArrayInCache(
  ndarrayCacheUrl: string,
  cacheScope = "tvmjs",
  cacheType = "cache"
): Promise<boolean> {
  let artifactCache: ArtifactCacheTemplate;
  if (cacheType.toLowerCase() === "cache") {
    artifactCache = new ArtifactCache(cacheScope);
  } else if (cacheType.toLowerCase() == "indexeddb") {
    artifactCache = new ArtifactIndexedDBCache(cacheScope);
  } else {
    console.error("Unsupported cacheType: " + cacheType + ", using default ArtifactCache.");
    artifactCache = new ArtifactCache(cacheScope);
  }
  const jsonUrl = new URL("ndarray-cache.json", ndarrayCacheUrl).href;
  const hasJsonUrlInCache = await artifactCache.hasAllKeys([jsonUrl]);
  if (!hasJsonUrlInCache) {
    return false;
  }
  let list = await artifactCache.fetchWithCache(jsonUrl, "json");
  list = list["records"] as Array<NDArrayShardEntry>;
  return await artifactCache.hasAllKeys(list.map(key => new URL(key.dataPath, ndarrayCacheUrl).href));
}


/**
 * Given cacheUrl, search up items to delete based on cacheUrl/ndarray-cache.json
 *
 * @param cacheUrl The cacheUrl for the items
 * @param cacheScope The scope identifier of the cache
 * @param cacheType The type of the cache: "cache" or "indexedDB"
 */
export async function deleteNDArrayCache(
  cacheUrl: string,
  cacheScope = "tvmjs",
  cacheType = "cache"
) {
  let artifactCache: ArtifactCacheTemplate;
  if (cacheType.toLowerCase() === "cache") {
    artifactCache = new ArtifactCache(cacheScope);
  } else if (cacheType.toLowerCase() == "indexeddb") {
    artifactCache = new ArtifactIndexedDBCache(cacheScope);
  } else {
    console.error("Unsupported cacheType: " + cacheType + ", using default ArtifactCache.");
    artifactCache = new ArtifactCache(cacheScope);
  }
  const jsonUrl = new URL("ndarray-cache.json", cacheUrl).href;
  const list = await artifactCache.fetchWithCache(jsonUrl, "json");
  const arrayentry = list["records"] as Array<NDArrayShardEntry>;
  const processShard = async (i: number) => {
    const dataUrl = new URL(arrayentry[i].dataPath, cacheUrl).href;
    await artifactCache.deleteInCache(dataUrl);
  }
  await Promise.all(arrayentry.map((_, index) => processShard(index)));
}

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
   * @return The data object (i.e. users do not need to call `.json()` or `.arraybuffer()`).
   * 
   * @note This is an async function.
   */
  fetchWithCache(url: string, storetype?: string): Promise<any>;

  /**
   * Fetch data from url and add into cache. If already exists in cache, should return instantly.
   *
   * @param url: The url to the data to be cached.
   * @param storetype: Only applies to `ArtifactIndexedDBCache`. Since `indexedDB` stores the actual
   * data rather than a request, we specify `storagetype`. There are two options:
   * 1. "json": IndexedDB stores `fetch(url).json()`
   * 2. "arraybuffer": IndexedDB stores `fetch(url).arrayBuffer()`
   * 
   * @note This is an async function.
   */
  addToCache(url: string, storetype?: string): Promise<void>;

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

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
   * fetch key url from cache, optional storetype for IndexedDB
   *
   * storagetype for indexedDB have two options:
   * @param url: return a json object
   * 2. arraybuffer: return an arraybuffer object
   */
  fetchWithCache(url: string, storetype?: string);

  /**
   * add key url to cache, optional storetype for IndexedDB
   *
   * storagetype for indexedDB have two options:
   * 1. json: return a json object
   * 2. arraybuffer: return an arraybuffer object
   *
   * returns the response or the specified stored object
   * for reduced database transaction
   */
  addToCache(url: string, storetype?: string): Promise<any>;
  /**
   * check if cache has all keys in Cache
   */
  hasAllKeys(keys: string[]);

  /**
   * Delete url in cache if url exists
   */
  deleteInCache(url: string);
}

/*
    Common Interface for the artifact cache
*/
export interface ArtifactCacheTemplate {
    /**
     * fetch key url from cache
     */
    fetchWithCache(url: string);

    /**
     * check if cache has all keys in Cache
     */
    hasAllKeys(keys: string[]);

    /**
     * Delete url in cache if url exists
     */
    deleteInCache(url: string);
}

import { useQuery } from "@tanstack/react-query";

import { fetchSearchCatalog } from "../../api/client";
import { SearchForm } from "./SearchForm";
import { SearchResults } from "./SearchResults";
import { buildVolumeOptions } from "./search-model";
import { usePassageSearch } from "./usePassageSearch";

export function SearchPage() {
  const catalog = useQuery({
    queryKey: ["search-catalog"],
    queryFn: ({ signal }) => fetchSearchCatalog(signal),
    staleTime: Infinity,
    retry: 1
  });
  const search = usePassageSearch();
  const volumeOptions = buildVolumeOptions(catalog.data);

  return (
    <>
      <SearchForm
        catalog={catalog.data}
        volumeOptions={volumeOptions}
        isFetching={search.isFetching}
        isSearching={search.isSearching}
        verificationError={search.verification.error}
        turnstileResetKey={search.verification.resetKey}
        onTurnstileToken={search.verification.onToken}
        onSubmit={search.submit}
      />
      <SearchResults
        results={search.results}
        onShowMore={search.showMore}
      />
    </>
  );
}

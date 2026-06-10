from pathlib import Path

from src.core.search import optimized_recursive_search, order_path_list


class TestOrderPathList:
    def test_sorted_by_plot_id(self):
        paths = ["/a/plot_5.parquet", "/a/plot_2.parquet", "/a/plot_10.parquet"]
        result = order_path_list(paths)
        assert result == ["/a/plot_2.parquet", "/a/plot_5.parquet", "/a/plot_10.parquet"]

    def test_no_plot_ids_returns_same_order(self):
        paths = ["/a/file.parquet", "/a/other.parquet"]
        result = order_path_list(paths)
        assert result == ["/a/file.parquet", "/a/other.parquet"]

    def test_mixed_plot_and_non_plot(self):
        paths = ["/a/plot_3.parquet", "/a/nomatch.parquet", "/a/plot_1.parquet"]
        result = order_path_list(paths)
        assert result == [
            "/a/plot_1.parquet",
            "/a/plot_3.parquet",
            "/a/nomatch.parquet",
        ]

    def test_empty_list(self):
        result = order_path_list([])
        assert result == []

    def test_single_entry(self):
        result = order_path_list(["/a/plot_7.parquet"])
        assert result == ["/a/plot_7.parquet"]

    def test_non_integer_plot_id_goes_to_rest(self):
        paths = ["/a/plot_abc.parquet", "/a/plot_1.parquet"]
        result = order_path_list(paths)
        assert result == ["/a/plot_1.parquet", "/a/plot_abc.parquet"]


class TestOptimizedRecursiveSearch:
    FOLDERS = ["", "metashape", "products_uav_data", "output", "extract", "polygon_df"]

    def _create_tree(self, base: Path) -> None:
        (base / "metashape").mkdir(parents=True, exist_ok=True)
        (base / "week1").mkdir(parents=True, exist_ok=True)
        (base / "random_week2_stuff").mkdir(parents=True, exist_ok=True)
        (base / "other").mkdir(parents=True, exist_ok=True)

        (base / "metashape" / "plot_1.parquet").touch()
        (base / "metashape" / "plot_2.parquet").touch()
        (base / "week1" / "plot_3.parquet").touch()
        (base / "week1" / "plot_4.parquet").touch()
        (base / "random_week2_stuff" / "plot_5.parquet").touch()
        (base / "other" / "plot_6.parquet").touch()

    def test_finds_files_in_subdirs(self, tmp_path):
        self._create_tree(tmp_path)
        results = optimized_recursive_search(
            self.FOLDERS, "plot_", start_dir=str(tmp_path), remove_unkwown=False
        )
        all_files = [f for files in results.values() for f in files]
        assert len(all_files) >= 5

    def test_remove_unknown_true_excludes_unknown(self, tmp_path):
        self._create_tree(tmp_path)
        results = optimized_recursive_search(
            self.FOLDERS, "plot_", start_dir=str(tmp_path), remove_unkwown=True
        )
        assert "unknown" not in results

    def test_remove_unknown_false_includes_unknown(self, tmp_path):
        self._create_tree(tmp_path)
        results = optimized_recursive_search(
            self.FOLDERS, "plot_", start_dir=str(tmp_path), remove_unkwown=False
        )
        # The "other" directory has no week in path and no folder match,
        # but "other" doesn't match the FOLDER patterns either.
        # Since "" matches everything, "other" folder will be checked.
        # Because it has no week ID, it falls to "unknown".
        # So with remove_unkwown=False, "unknown" should be present.
        assert "unknown" in results

    def test_extracts_week_ids(self, tmp_path):
        self._create_tree(tmp_path)
        results = optimized_recursive_search(self.FOLDERS, "plot_", start_dir=str(tmp_path))
        assert "week1" in results
        assert "week2" in results

    def test_search_directory_directly(self, tmp_path):
        d = tmp_path / "testdir"
        d.mkdir()
        (d / "data_plot_1.parquet").touch()
        (d / "data_plot_2.parquet").touch()
        (d / "unrelated.txt").touch()

        from src.core.search import search_directory

        matches = search_directory(str(d), "plot_")
        assert matches is not None
        assert len(matches) == 2
        for m in matches:
            assert m.endswith(".parquet")

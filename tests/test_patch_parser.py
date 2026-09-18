"""Tests for AST-based unified diff parsing (kg_construction#75).

PatchParser resolves changed functions/classes by parsing the pre-patch
source's real AST and mapping each changed line number against real
lineno/end_lineno ranges, rather than inferring boundaries from the diff
text's own (small, variable) context window. This replaces an earlier
hunk-scanning approach that caused four distinct bugs over time (#14, #43,
#63, #71) from the same root cause -- see patch.py's module docstring.

Each test constructs BOTH a patch and its corresponding pre-patch source,
since the new API requires the latter to resolve real ranges.
"""

from kg_construction.extraction.patch import (
    PatchParser,
    is_newly_created_file,
    reconstruct_post_patch_source,
)


class TestPatchParser:
    """Test PatchParser.extract_changed_functions()."""

    def test_extract_function_body_modified(self):
        source = (
            "def send(self, method, url, **kwargs):\n"
            "    response = self._request(url)\n"
            "    return response\n"
        )
        patch = """--- a/requests/sessions.py
+++ b/requests/sessions.py
@@ -1,3 +1,4 @@
 def send(self, method, url, **kwargs):
     response = self._request(url)
-    return response
+    return response  # cached
"""
        changed = PatchParser.extract_changed_functions(patch, 'requests/sessions.py', source)
        assert 'send' in changed

    def test_extract_class_method_modified(self):
        source = (
            "class NewSession:\n"
            "    def __init__(self):\n"
            "        pass\n"
            "\n"
            "    def send(self):\n"
            "        return 1\n"
        )
        patch = """--- a/requests/sessions.py
+++ b/requests/sessions.py
@@ -4,3 +4,3 @@
     def send(self):
-        return 1
+        return 2
"""
        changed = PatchParser.extract_changed_functions(patch, 'requests/sessions.py', source)
        assert 'send' in changed
        assert 'NewSession' not in changed  # only the method's own body changed

    def test_ignore_changes_in_other_files(self):
        """Only code_file's own hunks are considered -- a multi-file patch
        must not pull in changes from a different file's hunks.
        """
        source = "def target_function():\n    return 1\n"
        patch = """--- a/requests/sessions.py
+++ b/requests/sessions.py
@@ -1,2 +1,2 @@
 def target_function():
-    return 1
+    return 2
--- a/requests/adapters.py
+++ b/requests/adapters.py
@@ -1,2 +1,2 @@
 def other_function():
-    return 1
+    return 2
"""
        changed = PatchParser.extract_changed_functions(patch, 'requests/sessions.py', source)
        assert changed == {'target_function'}

    def test_multi_file_patch_final_hunk(self):
        """A target file that isn't last in a multi-file patch is still
        fully processed, not just its first hunk.
        """
        source = "def first_change():\n    return 1\n\n\ndef also_here():\n    return 2\n"
        patch = """--- a/requests/sessions.py
+++ b/requests/sessions.py
@@ -1,2 +1,2 @@
 def first_change():
-    return 1
+    return 10
@@ -5,2 +5,2 @@
 def also_here():
-    return 2
+    return 20
--- a/requests/adapters.py
+++ b/requests/adapters.py
@@ -1,2 +1,2 @@
 def second_change():
-    return 1
+    return 2
"""
        changed = PatchParser.extract_changed_functions(patch, 'requests/sessions.py', source)
        assert changed == {'first_change', 'also_here'}

    def test_empty_patch(self):
        source = "def f():\n    return 1\n"
        changed = PatchParser.extract_changed_functions('', 'requests/sessions.py', source)
        assert len(changed) == 0

    def test_patch_no_target_file(self):
        source = "def f():\n    return 1\n"
        patch = """--- a/requests/adapters.py
+++ b/requests/adapters.py
@@ -1,2 +1,2 @@
 def some_function():
-    return 1
+    return 2
"""
        changed = PatchParser.extract_changed_functions(patch, 'requests/sessions.py', source)
        assert len(changed) == 0

    def test_decorator_change_attributed_to_decorated_function(self):
        """A changed decorator line, though outside the def line's own
        ast.lineno, must still be attributed to the function it decorates --
        decorator_list's own line is included in the function's range.
        """
        source = '@app.route("/old")\ndef handler():\n    pass\n'
        patch = """--- a/app.py
+++ b/app.py
@@ -1,2 +1,2 @@
-@app.route("/old")
+@app.route("/new")
 def handler():
"""
        changed = PatchParser.extract_changed_functions(patch, 'app.py', source)
        assert 'handler' in changed

    def test_wide_context_does_not_sweep_in_unmodified_sibling(self):
        """Regression test for issue #43: an unmodified neighboring
        function's def line appearing in the same hunk (as context) must
        not be reported as changed -- resolution is by real line ranges,
        not by what's merely visible in the hunk.
        """
        source = (
            "def delete(self, url, **kwargs):\n"
            "    kwargs.setdefault('allow_redirects', True)\n"
            "    return self.request('DELETE', url, **kwargs)\n"
            "\n"
            "def send(self, request, **kwargs):\n"
            "    kwargs.setdefault('stream', self.stream)\n"
            "    kwargs.setdefault('verify', self.verify)\n"
            "    kwargs.setdefault('cert', self.cert)\n"
            "    kwargs.setdefault('proxies', self.proxies)\n"
            "\n"
            "    pass\n"
        )
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,10 +1,10 @@
 def delete(self, url, **kwargs):
     kwargs.setdefault('allow_redirects', True)
     return self.request('DELETE', url, **kwargs)

 def send(self, request, **kwargs):
     kwargs.setdefault('stream', self.stream)
     kwargs.setdefault('verify', self.verify)
     kwargs.setdefault('cert', self.cert)
-    kwargs.setdefault('proxies', self.proxies)
+    kwargs.setdefault('proxies', self.rebuild_proxies(request, self.proxies))

     pass
"""
        changed = PatchParser.extract_changed_functions(patch, 'mod.py', source)
        assert changed == {'send'}

    def test_body_change_detection_does_not_leak_into_siblings(self):
        source = (
            "def unrelated():\n    pass\n\n"
            "def target():\n    return 1\n\n"
            "def another_unrelated():\n    pass\n"
        )
        patch = """--- a/mod.py
+++ b/mod.py
@@ -4,2 +4,2 @@
 def target():
-    return 1
+    return 2
"""
        changed = PatchParser.extract_changed_functions(patch, 'mod.py', source)
        assert changed == {'target'}

    def test_context_only_def_with_unchanged_body_not_reported(self):
        """A def/class fully visible in the hunk as unchanged context must
        not be reported -- only its real line range, not mere visibility
        in the diff, decides attribution.
        """
        source = "def untouched():\n    return 1\n\ndef changed():\n    return 2\n"
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,5 +1,5 @@
 def untouched():
     return 1

 def changed():
-    return 2
+    return 3
"""
        changed = PatchParser.extract_changed_functions(patch, 'mod.py', source)
        assert changed == {'changed'}
        assert 'untouched' not in changed

    def test_change_deep_in_long_function_still_attributed_correctly(self):
        """kg_construction#60's original motivation: a change far enough
        into a long function that a small diff context window wouldn't
        show its def line at all. Real ast ranges make this a non-issue --
        no context-window dependency at all.
        """
        source = (
            "def long_function(value):\n"
            "    step_one = value\n"
            "    step_two = step_one + 1\n"
            "    step_three = step_two * 2\n"
            "    return step_three\n"
        )
        patch = """--- a/mod.py
+++ b/mod.py
@@ -4,2 +4,2 @@
-    return step_three
+    return step_three + 1
"""
        changed = PatchParser.extract_changed_functions(patch, 'mod.py', source)
        assert changed == {'long_function'}

    def test_change_outside_any_function_reports_nothing(self):
        """A changed line at module level, not inside any function/class,
        correctly reports no changed function (kg_construction#71's
        regression case, re-verified under the new AST-based approach).
        """
        source = "def untouched():\n    return 1\n# a module-level comment\n"
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,3 +1,3 @@
 def untouched():
     return 1
-# a module-level comment
+# a different module-level comment
"""
        changed = PatchParser.extract_changed_functions(patch, 'mod.py', source)
        assert changed == set()

    def test_real_pytest_multi_hunk_case_from_issue_71(self):
        """The exact real-world shape that surfaced #71: a changed line
        deep inside a function, followed later in the SAME hunk by an
        unrelated, unmodified sibling function's def line swept in as
        trailing context. Must not suppress or misattribute the real
        change -- confirmed directly against pytest-dev/pytest's own
        _showfixtures_main patch under the old hunk-scanning approach;
        re-verified here that the AST-based approach gets it right too,
        without needing any special-casing for this shape at all.
        """
        source = (
            "def real_target(x):\n"
            "    if x > 0:\n"
            "        return x\n"
            "\n"
            "\n"
            "def unrelated_sibling(y):\n"
            "    return y\n"
        )
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,3 +1,4 @@
 def real_target(x):
     if x > 0:
+        print("added")
         return x
"""
        changed = PatchParser.extract_changed_functions(patch, 'mod.py', source)
        assert changed == {'real_target'}


class TestExtractChangedFunctionsWithScope:
    """kg_construction#63: a changed method name can match more than one
    class' same-named method in the same file. extract_changed_functions_with_scope
    reports the enclosing class alongside each name so TestContextExtractor
    can disambiguate. Resolution is now by real line range -- two same-named
    methods on different classes have different ranges, so this ambiguity
    can't even arise structurally (unlike the old hunk-scanning approach,
    which needed an explicit class-hint mechanism to work around it).
    """

    def test_method_reports_its_enclosing_class(self):
        source = "class Widget:\n    def build(self):\n        return self.helper()\n"
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,3 +1,4 @@
 class Widget:
     def build(self):
         return self.helper()
+        # a comment
"""
        changed = PatchParser.extract_changed_functions_with_scope(patch, 'mod.py', source)
        assert ('build', 'Widget') in changed

    def test_no_class_information_reports_none(self):
        source = "def standalone():\n    return 1\n"
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,2 +1,3 @@
 def standalone():
     return 1
+    # comment
"""
        changed = PatchParser.extract_changed_functions_with_scope(patch, 'mod.py', source)
        assert ('standalone', None) in changed

    def test_class_scope_does_not_leak_to_module_level_function(self):
        source = (
            "class Widget:\n"
            "    def build(self):\n"
            "        return 1\n"
            "\n"
            "def standalone():\n"
            "    return 2\n"
        )
        patch = """--- a/mod.py
+++ b/mod.py
@@ -5,2 +5,3 @@
 def standalone():
     return 2
+    # comment
"""
        changed = PatchParser.extract_changed_functions_with_scope(patch, 'mod.py', source)
        assert ('standalone', None) in changed
        assert ('standalone', 'Widget') not in changed

    def test_two_classes_same_method_name_get_different_scopes(self):
        """The exact ambiguity #63 was found from: two unrelated classes
        each define a method with the same name. Both must be reported,
        each with its OWN correct enclosing class -- resolved here purely
        by real line range, no special ambiguity-handling needed.
        """
        source = (
            "class Alpha:\n"
            "    def aclose(self):\n"
            "        return 1\n"
            "\n"
            "class Beta:\n"
            "    def aclose(self):\n"
            "        return 2\n"
        )
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,7 +1,9 @@
 class Alpha:
     def aclose(self):
         return 1
+        # comment

 class Beta:
     def aclose(self):
         return 2
+        # comment
"""
        changed = PatchParser.extract_changed_functions_with_scope(patch, 'mod.py', source)
        assert ('aclose', 'Alpha') in changed
        assert ('aclose', 'Beta') in changed

    def test_nested_function_reports_no_class_even_inside_a_method(self):
        """kg_construction#74: a closure nested inside a method is
        attributed to the closure itself, not the enclosing method or
        class -- and correctly has no enclosing CLASS (its immediate
        enclosing scope is a function, not a class). The changed line sits
        in the middle of the closure's body (not at its very first/last
        line) so the change is unambiguously inside the closure alone, not
        at a boundary that could also plausibly belong to the enclosing
        method.
        """
        source = (
            "class Widget:\n"
            "    def build(self, items):\n"
            "        def sort_key(entry):\n"
            "            key = entry[1]\n"
            "            return key\n"
            "        return sorted(items, key=sort_key)\n"
        )
        patch = """--- a/mod.py
+++ b/mod.py
@@ -3,3 +3,4 @@
         def sort_key(entry):
             key = entry[1]
+            # comment
             return key
"""
        changed = PatchParser.extract_changed_functions_with_scope(patch, 'mod.py', source)
        assert ('sort_key', None) in changed
        assert ('build', 'Widget') not in changed

    def test_flat_extract_changed_functions_still_returns_bare_names(self):
        """extract_changed_functions (still used by resolve_target_function/
        build_baseline_context) must keep returning a flat Set[str]."""
        source = "class Widget:\n    def build(self):\n        return 1\n"
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,3 +1,4 @@
 class Widget:
     def build(self):
         return 1
+        # comment
"""
        changed = PatchParser.extract_changed_functions(patch, 'mod.py', source)
        assert changed == {'build'}


class TestReconstructPostPatchSource:
    """reconstruct_post_patch_source applies patch's hunks to
    pre_patch_source in memory (kg_construction#84's fix depends on this
    being an EXACT reconstruction, not an approximation -- a wrong
    reconstruction would silently feed bad source into ast.parse and
    produce wrong ranges). Verified directly against real GitHub patches
    applied via `git apply` as ground truth (see #84's PR description) --
    these are the synthetic, in-repo equivalents of that same check.
    """

    def test_single_hunk_addition_reconstructs_exactly(self):
        pre = "def a():\n    return 1\n\n\ndef b():\n    return 2\n"
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,6 +1,10 @@
 def a():
     return 1

+
+def new_func():
+    return 42

 def b():
     return 2
"""
        post = reconstruct_post_patch_source(pre, patch, 'mod.py')
        assert post == (
            "def a():\n    return 1\n\n\n"
            "def new_func():\n    return 42\n\n"
            "def b():\n    return 2\n"
        )

    def test_pure_removal_reconstructs_exactly(self):
        pre = "def a():\n    x = 1\n    y = 2\n    return x + y\n"
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,4 +1,3 @@
 def a():
     x = 1
-    y = 2
-    return x + y
+    return x + 1
"""
        post = reconstruct_post_patch_source(pre, patch, 'mod.py')
        assert post == "def a():\n    x = 1\n    return x + 1\n"

    def test_multiple_hunks_reconstruct_exactly(self):
        pre = (
            "def a():\n    return 1\n\n"
            "def b():\n    return 2\n\n"
            "def c():\n    return 3\n"
        )
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,2 +1,2 @@
 def a():
-    return 1
+    return 100
@@ -7,2 +7,2 @@
 def c():
-    return 3
+    return 300
"""
        post = reconstruct_post_patch_source(pre, patch, 'mod.py')
        assert post == (
            "def a():\n    return 100\n\n"
            "def b():\n    return 2\n\n"
            "def c():\n    return 300\n"
        )

    def test_no_hunks_for_code_file_returns_none(self):
        pre = "def a():\n    return 1\n"
        patch = """--- a/other.py
+++ b/other.py
@@ -1,2 +1,2 @@
 def x():
-    return 1
+    return 2
"""
        assert reconstruct_post_patch_source(pre, patch, 'mod.py') is None

    def test_trailing_newline_artifact_is_not_treated_as_a_context_line(self):
        # str.split('\n') on a real, newline-terminated patch always adds
        # a trailing '' element. Without stripping it, that artifact gets
        # checked as a genuine blank context line against pre_lines and
        # (correctly, but wrongly here) fails validation, since it has no
        # real counterpart in the source (kg_construction#128).
        pre = "def a():\n    return 1\n"
        patch = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def a():\n"
            "-    return 1\n"
            "+    return 2\n"
        )
        assert patch.endswith("\n")
        post = reconstruct_post_patch_source(pre, patch, 'mod.py')
        assert post == "def a():\n    return 2\n"

    def test_context_line_mismatch_returns_none(self):
        # pre_patch_source doesn't actually match what this patch assumes
        # (a context line the diff claims is 'y = 2' isn't really there) --
        # must be detected and rejected, not silently reconstructed wrong
        # (kg_construction#128).
        pre = "def a():\n    x = 1\n    z = 3\n    return x\n"
        patch = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -1,4 +1,4 @@\n"
            " def a():\n"
            "     x = 1\n"
            "     y = 2\n"
            "-    return x\n"
            "+    return x + 1\n"
        )
        assert reconstruct_post_patch_source(pre, patch, 'mod.py') is None

    def test_removed_line_mismatch_returns_none(self):
        pre = "def a():\n    return 1\n"
        patch = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def a():\n"
            "-    return 999\n"
            "+    return 2\n"
        )
        assert reconstruct_post_patch_source(pre, patch, 'mod.py') is None

    def test_out_of_order_hunk_returns_none(self):
        # A well-formed diff's hunks are always ascending and
        # non-overlapping. A second hunk starting behind where the first
        # hunk already advanced pre_idx means pre_patch_source doesn't
        # correspond to what this patch was generated against -- the same
        # mismatch as a forward overshoot past pre_lines' length, just in
        # the other direction (kg_construction#128).
        pre = "a\nb\nc\nd\ne\n"
        patch = (
            "--- a/mod.py\n"
            "+++ b/mod.py\n"
            "@@ -4,1 +4,1 @@\n"
            "-d\n"
            "+D\n"
            "@@ -1,1 +1,1 @@\n"
            "-a\n"
            "+A\n"
        )
        assert reconstruct_post_patch_source(pre, patch, 'mod.py') is None


class TestWhollyNewFunctionsAndClasses:
    """kg_construction#84: a wholly new top-level function/class doesn't
    exist anywhere in the pre-patch source, so it has no range to be
    attributed against there. Resolved by also checking the reconstructed
    POST-patch source's own ranges and unioning both result sets.
    """

    def test_new_module_level_function_between_two_others_is_detected(self):
        """The exact real-world shape that surfaced #84 (a real sympy
        patch, sympy__sympy-21055): a new function inserted in the
        blank-line gap between two existing ones -- that gap isn't inside
        EITHER neighboring function's pre-patch range, so there's nothing
        there to attribute the insertion to on the pre-patch side alone.
        """
        source = (
            "def before():\n"
            "    return 1\n"
            "\n"
            "\n"
            "def after():\n"
            "    return 2\n"
        )
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,6 +1,10 @@
 def before():
     return 1

+
+def new_function():
+    return 42
+

 def after():
     return 2
"""
        changed = PatchParser.extract_changed_functions(patch, 'mod.py', source)
        assert 'new_function' in changed

    def test_new_method_inserted_immediately_before_an_existing_one(self):
        """kg_construction#84's follow-up finding: a new method inserted
        directly before an existing sibling method must not be
        misattributed to that sibling merely because the insertion point
        lands on the sibling's own def line (which is trivially "inside"
        the sibling's own range).
        """
        source = (
            "class Printer:\n"
            "    def existing_method(self, x):\n"
            "        return x\n"
        )
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,3 +1,6 @@
 class Printer:
+    def new_method(self, x):
+        return x * 2
+
     def existing_method(self, x):
         return x
"""
        changed = PatchParser.extract_changed_functions_with_scope(patch, 'mod.py', source)
        assert ('new_method', 'Printer') in changed
        assert ('existing_method', 'Printer') not in changed

    def test_new_method_added_at_end_of_existing_class(self):
        """A new method appended after the class's existing method(s) --
        the insertion point falls at/after the end of the pre-patch
        class's own range, so it also has no home there."""
        source = (
            "class Widget:\n"
            "    def build(self):\n"
            "        return 1\n"
        )
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,3 +1,6 @@
 class Widget:
     def build(self):
         return 1
+
+    def new_method(self):
+        return 2
"""
        changed = PatchParser.extract_changed_functions_with_scope(patch, 'mod.py', source)
        assert ('new_method', 'Widget') in changed

    def test_new_class_between_two_others_is_detected(self):
        source = (
            "class Alpha:\n"
            "    pass\n"
            "\n"
            "\n"
            "class Beta:\n"
            "    pass\n"
        )
        patch = """--- a/mod.py
+++ b/mod.py
@@ -2,5 +2,10 @@ class Alpha:
     pass


+class NewClass:
+    def method(self):
+        return 1
+
+
 class Beta:
     pass
"""
        changed = PatchParser.extract_changed_functions(patch, 'mod.py', source)
        assert 'NewClass' in changed

    def test_ordinary_modification_is_not_duplicated_by_post_patch_pass(self):
        """An ordinary modification (no new function/class involved) must
        not gain a spurious extra entry just because both the pre- and
        post-patch passes independently resolve it -- the union must
        still produce exactly the one real changed function.
        """
        source = "def target():\n    return 1\n"
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,2 +1,2 @@
 def target():
-    return 1
+    return 2
"""
        changed = PatchParser.extract_changed_functions(patch, 'mod.py', source)
        assert changed == {'target'}


class TestIsNewlyCreatedFile:
    """kg_construction#93: a file the patch itself creates (git's 'new file
    mode' header) genuinely doesn't exist at base_commit -- callers must
    detect this BEFORE trying to fetch pre-patch source, rather than
    treating the fetch failure as an error.
    """

    def test_detects_new_file_mode_header(self):
        patch = """diff --git a/pkg/new_mod.py b/pkg/new_mod.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/pkg/new_mod.py
@@ -0,0 +1,2 @@
+def foo():
+    pass
"""
        assert is_newly_created_file(patch, 'pkg/new_mod.py') is True

    def test_ordinary_modification_is_not_new_file(self):
        patch = """diff --git a/pkg/existing.py b/pkg/existing.py
index 1234567..89abcde 100644
--- a/pkg/existing.py
+++ b/pkg/existing.py
@@ -1,2 +1,2 @@
 def foo():
-    return 1
+    return 2
"""
        assert is_newly_created_file(patch, 'pkg/existing.py') is False

    def test_new_file_mode_for_a_different_file_does_not_leak(self):
        """A multi-file patch where SOME OTHER file is new must not mark
        code_file itself as new.
        """
        patch = """diff --git a/pkg/brand_new.py b/pkg/brand_new.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/pkg/brand_new.py
@@ -0,0 +1 @@
+x = 1
diff --git a/pkg/existing.py b/pkg/existing.py
index 1234567..89abcde 100644
--- a/pkg/existing.py
+++ b/pkg/existing.py
@@ -1,2 +1,2 @@
 def foo():
-    return 1
+    return 2
"""
        assert is_newly_created_file(patch, 'pkg/existing.py') is False
        assert is_newly_created_file(patch, 'pkg/brand_new.py') is True


class TestWhollyNewFileResolution:
    """kg_construction#93 end-to-end: with no pre-patch source at all
    (pre_patch_source=""), a wholly new file's own functions/classes must
    still resolve correctly via the existing post-patch reconstruction path
    (kg_construction#84) -- an empty pre-patch source is just the smallest
    possible case of "nothing here to attribute an insertion to yet".
    """

    def test_new_file_functions_resolve_from_empty_pre_patch_source(self):
        patch = """diff --git a/pkg/new_mod.py b/pkg/new_mod.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/pkg/new_mod.py
@@ -0,0 +1,5 @@
+def first():
+    return 1
+
+def second():
+    return 2
"""
        changed = PatchParser.extract_changed_functions(patch, 'pkg/new_mod.py', "")
        assert changed == {'first', 'second'}

    def test_new_file_class_resolves_from_empty_pre_patch_source(self):
        patch = """diff --git a/pkg/new_mod.py b/pkg/new_mod.py
new file mode 100644
index 0000000..1234567
--- /dev/null
+++ b/pkg/new_mod.py
@@ -0,0 +1,3 @@
+class SimpleImputer:
+    def transform(self, x):
+        return x
"""
        changed = PatchParser.extract_changed_functions_with_scope(patch, 'pkg/new_mod.py', "")
        assert ('SimpleImputer', None) in changed
        assert ('transform', 'SimpleImputer') in changed


class TestClassLevelAnchorRedundancy:
    """kg_construction#85's discovery: an insertion anchored to the blank
    line right after a method's closing line has no smaller range to match
    (the method's own range already ended), so it matches the ENCLOSING
    CLASS -- previously harmless noise since nothing looked up class nodes,
    but once find_class_by_name existed this spuriously added the whole
    class as an extra seed alongside the real, correctly-matched method.
    """

    def test_insertion_right_after_a_changed_methods_body_does_not_also_match_the_class(self):
        source = (
            "class Widget:\n"
            "    def build(self):\n"
            "        return self.helper()\n"
            "\n"
            "    def helper(self):\n"
            "        return 42\n"
            "\n"
            "    def unrelated(self):\n"
            "        return 0\n"
        )
        patch = """--- a/mod.py
+++ b/mod.py
@@ -2,3 +2,3 @@
     def build(self):
-        return self.helper()
+        return self.helper() + 1
"""
        changed = PatchParser.extract_changed_functions_with_scope(patch, 'mod.py', source)
        assert changed == {('build', 'Widget')}

    def test_genuine_class_body_attribute_change_still_matches_the_class(self):
        """A real class-body-level change (no method touched at all) must
        still resolve to the class itself -- the redundancy filter only
        drops a class-level match when some OTHER entry in the same result
        is a method of that exact class.
        """
        source = (
            "class Widget:\n"
            "    \"\"\"doc\"\"\"\n"
            "\n"
            "    def build(self):\n"
            "        return 1\n"
        )
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,2 +1,3 @@
 class Widget:
+    template_name = "x"
     \"\"\"doc\"\"\"
"""
        changed = PatchParser.extract_changed_functions_with_scope(patch, 'mod.py', source)
        assert ('Widget', None) in changed

    def test_new_class_with_new_method_reports_both(self):
        """A genuinely new class AND its own new method (both resolved via
        the trustworthy post-patch pass, not the ambiguous anchor pass)
        must both be reported -- the redundancy filter must not conflate
        this with the anchor-noise case.
        """
        source = "class Existing:\n    pass\n"
        patch = """--- a/mod.py
+++ b/mod.py
@@ -1,2 +1,6 @@
 class Existing:
     pass
+
+class NewClass:
+    def new_method(self):
+        return 1
"""
        changed = PatchParser.extract_changed_functions_with_scope(patch, 'mod.py', source)
        assert ('NewClass', None) in changed
        assert ('new_method', 'NewClass') in changed

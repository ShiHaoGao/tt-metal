// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "SourceArchive.hpp"
#include "jit_build/depend.hpp"
#include <gtest/gtest.h>
#include <fstream>
#include <unistd.h>
namespace tt::tt_metal::native_firmware_bundle {
namespace {
namespace fs = std::filesystem;
class SupplierSources : public ::testing::Test {
protected:
    void SetUp() override {
        std::string pattern = (fs::temp_directory_path() / "supplier_sources_XXXXXX").string();
        auto* made = mkdtemp(pattern.data());
        ASSERT_NE(made, nullptr);
        root = made;
        fs::create_directory(root / "sdk");
        std::ofstream(root / "sdk/kernel.cpp") << "original source";
    }
    void TearDown() override { if (!root.empty()) fs::remove_all(root); }
    fs::path root;
};
TEST_F(SupplierSources, PreservesSourceAndGeneratedBytesInAtomicPublication) {
    Publication publication(root / "output");
    SourceArchive sources(publication, root / "sdk");
    fs::create_directories(publication.cache());
    std::ofstream(publication.cache() / "generated.h") << "generated header";
    EXPECT_EQ(sources.capture(root / "sdk/kernel.cpp"), "provenance/source/kernel.cpp");
    EXPECT_EQ(sources.capture(publication.cache() / "generated.h"), "provenance/generated/generated.h");
    EXPECT_EQ(sources.capture(root / "sdk/kernel.cpp"), "provenance/source/kernel.cpp");
    EXPECT_EQ(sources.records.size(), 2);
    EXPECT_NO_THROW(sources.verify());
    publication.commit();
    std::ifstream file(root / "output/provenance/generated/generated.h");
    EXPECT_EQ(std::string(std::istreambuf_iterator<char>(file), {}), "generated header");
}
TEST_F(SupplierSources, RejectsSourceMutationEvenWhenItIsNotRecaptured) {
    Publication publication(root / "output");
    SourceArchive sources(publication, root / "sdk");
    sources.capture(root / "sdk/kernel.cpp");
    std::ofstream(root / "sdk/kernel.cpp") << "modified source";
    EXPECT_THROW(sources.verify(), std::runtime_error);
    EXPECT_THROW(sources.capture(root / "sdk/kernel.cpp"), std::runtime_error);
    EXPECT_FALSE(fs::exists(root / "output"));
}
TEST_F(SupplierSources, RejectsMissingRequiredSourceTree) {
    Publication publication(root / "output");
    SourceArchive sources(publication, root / "sdk");
    EXPECT_THROW(sources.capture_tree(root / "sdk/missing"), std::runtime_error);
}

class SupplierDependencies : public SupplierSources {
protected:
    tt::jit_build::TargetRecipe recipe;
    fs::path image;
    fs::path header;
    fs::path link_input;
    void prepare(Publication& publication) {
        image = publication.cache() / "brisc/brisc.elf";
        fs::create_directories(image.parent_path());
        header = publication.cache() / "generated.h";
        link_input = publication.cache() / "brisc_weakened.elf";
        std::ofstream(header) << "generated header";
        std::ofstream(link_input) << "link input";
        std::ofstream(image) << "linked image";
        recipe.target_name = "brisc";
        recipe.srcs = {(root / "sdk/kernel.cpp").string()};
        recipe.objs = {"kernel.o"};
        record(image.parent_path() / "kernel.o", {root / "sdk/kernel.cpp", header});
        record(image, {link_input});
    }
    static void record(const fs::path& output, std::initializer_list<fs::path> paths) {
        tt::jit_build::ParsedDependencies deps;
        for (const auto& path : paths) deps[output.string()].push_back(path.string());
        std::ofstream record(output.string() + ".dephash");
        tt::jit_build::clear_file_hash_cache();
        tt::jit_build::write_dependency_hashes(deps, output.parent_path().string(), output.string(), record);
        ASSERT_TRUE(record.good());
    }
};

TEST_F(SupplierDependencies, CapturesObjectHeadersAndLinkInputsWithoutAnyDepfiles) {
    Publication publication(root / "output"); prepare(publication);
    SourceArchive sources(publication, root / "sdk");
    EXPECT_FALSE(fs::exists(image.parent_path() / "kernel.d"));
    sources.capture_recipe(recipe, image);
    EXPECT_TRUE(fs::is_regular_file(publication.payload() / "provenance/source/kernel.cpp"));
    EXPECT_TRUE(fs::is_regular_file(publication.payload() / "provenance/generated/generated.h"));
    EXPECT_TRUE(fs::is_regular_file(publication.payload() / "provenance/generated/brisc_weakened.elf"));
    EXPECT_TRUE(fs::is_regular_file(publication.payload() / "provenance/generated/brisc/kernel.o.dephash"));
    EXPECT_TRUE(fs::is_regular_file(publication.payload() / "provenance/generated/brisc/brisc.elf.dephash"));
    EXPECT_NO_THROW(sources.verify());
}

TEST_F(SupplierDependencies, RejectsSourceChangedAfterCompileBeforeFirstCaptureDespiteCachedMetadata) {
    Publication publication(root / "output"); prepare(publication);
    SourceArchive sources(publication, root / "sdk");
    const auto path = root / "sdk/kernel.cpp";
    const auto time = fs::last_write_time(path);
    ASSERT_TRUE(tt::jit_build::dependencies_up_to_date_file((image.parent_path() / "kernel.o.dephash").string()));
    std::ofstream(path) << "modified source";
    fs::last_write_time(path, time);
    EXPECT_THROW(sources.capture_recipe(recipe, image), std::runtime_error);
    EXPECT_FALSE(fs::exists(root / "output"));
}

TEST_F(SupplierDependencies, RequiresEveryRecipeObjectAndLinkDependencyRecord) {
    Publication publication(root / "output"); prepare(publication);
    SourceArchive sources(publication, root / "sdk");
    recipe.objs.push_back("missing.o");
    EXPECT_THROW(sources.capture_recipe(recipe, image), std::runtime_error);
    recipe.objs.pop_back();
    fs::remove(image.string() + ".dephash");
    EXPECT_THROW(sources.capture_recipe(recipe, image), std::runtime_error);
    EXPECT_FALSE(fs::exists(root / "output"));
}

TEST_F(SupplierDependencies, RejectsEmptyOrMalformedDependencyRecord) {
    Publication publication(root / "output"); prepare(publication);
    SourceArchive sources(publication, root / "sdk");
    std::ofstream(image.string() + ".dephash") << "";
    EXPECT_THROW(sources.capture_recipe(recipe, image), std::runtime_error);
    std::ofstream(image.string() + ".dephash") << "\"/some/file\" not-a-hash\n";
    EXPECT_THROW(sources.capture_recipe(recipe, image), std::runtime_error);
}

TEST_F(SupplierDependencies, RejectsLaterGeneratedInputMutationAtFinalVerification) {
    Publication publication(root / "output"); prepare(publication);
    SourceArchive sources(publication, root / "sdk");
    sources.capture_recipe(recipe, image);
    std::ofstream(header) << "changed generated header";
    EXPECT_THROW(sources.verify(), std::runtime_error);
    EXPECT_FALSE(fs::exists(root / "output"));
}
}
}

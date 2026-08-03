// Verifies the sigv4 module against AWS's published "get-vanilla" SigV4 test
// vector (aws-sig-v4-test-suite), independently re-derived and cross-checked
// against a from-scratch Python HMAC chain before being hardcoded here.
// Run with: pio test -e native
#include <unity.h>

#include "sigv4.h"

static const char *kAccessKey = "AKIDEXAMPLE";
static const char *kSecretKey = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
static const char *kDateStamp = "20150830";
static const char *kAmzDate = "20150830T123600Z";
static const char *kRegion = "us-east-1";
static const char *kService = "service";

static const char *kExpectedCanonicalRequest =
    "GET\n"
    "/\n"
    "\n"
    "host:example.amazonaws.com\n"
    "x-amz-date:20150830T123600Z\n"
    "\n"
    "host;x-amz-date\n"
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";

static const char *kExpectedStringToSign =
    "AWS4-HMAC-SHA256\n"
    "20150830T123600Z\n"
    "20150830/us-east-1/service/aws4_request\n"
    "bb579772317eb040ac9ed261061d46c1f17a8133879d6129b6e1c25292927e63";

static const char *kExpectedSignature =
    "5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31";

static const char *kExpectedAuthorizationHeader =
    "AWS4-HMAC-SHA256 "
    "Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, "
    "SignedHeaders=host;x-amz-date, "
    "Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31";

void test_sha256_empty_string(void) {
  TEST_ASSERT_EQUAL_STRING(
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      sigv4::sha256_hex("").c_str());
}

void test_canonical_request_matches_aws_vector(void) {
  std::string signed_headers;
  const std::string payload_hash = sigv4::sha256_hex("");
  const std::string creq = sigv4::canonical_request(
      "GET", "/", "",
      {{"host", "example.amazonaws.com"}, {"x-amz-date", kAmzDate}},
      payload_hash, signed_headers);

  TEST_ASSERT_EQUAL_STRING("host;x-amz-date", signed_headers.c_str());
  TEST_ASSERT_EQUAL_STRING(kExpectedCanonicalRequest, creq.c_str());
}

void test_string_to_sign_matches_aws_vector(void) {
  const std::string sts = sigv4::string_to_sign(
      kAmzDate, kDateStamp, kRegion, kService, kExpectedCanonicalRequest);
  TEST_ASSERT_EQUAL_STRING(kExpectedStringToSign, sts.c_str());
}

void test_full_signature_matches_aws_vector(void) {
  uint8_t signing_key[32];
  sigv4::derive_signing_key(kSecretKey, kDateStamp, kRegion, kService,
                             signing_key);
  const std::string signature =
      sigv4::sign_hex(signing_key, kExpectedStringToSign);
  TEST_ASSERT_EQUAL_STRING(kExpectedSignature, signature.c_str());
}

void test_authorization_header_matches_aws_vector(void) {
  const std::string header = sigv4::authorization_header(
      kAccessKey, kDateStamp, kRegion, kService, "host;x-amz-date",
      kExpectedSignature);
  TEST_ASSERT_EQUAL_STRING(kExpectedAuthorizationHeader, header.c_str());
}

// Headers passed out of order must still sort correctly into the canonical
// form (MinIO PUT requests build {x-amz-date, x-amz-content-sha256, host} in
// whatever order the caller assembles them).
void test_canonical_request_sorts_unordered_headers(void) {
  std::string signed_headers;
  const std::string creq = sigv4::canonical_request(
      "GET", "/", "",
      {{"x-amz-date", kAmzDate}, {"host", "example.amazonaws.com"}},
      sigv4::sha256_hex(""), signed_headers);
  TEST_ASSERT_EQUAL_STRING("host;x-amz-date", signed_headers.c_str());
  TEST_ASSERT_EQUAL_STRING(kExpectedCanonicalRequest, creq.c_str());
}

int main(int argc, char **argv) {
  UNITY_BEGIN();
  RUN_TEST(test_sha256_empty_string);
  RUN_TEST(test_canonical_request_matches_aws_vector);
  RUN_TEST(test_string_to_sign_matches_aws_vector);
  RUN_TEST(test_full_signature_matches_aws_vector);
  RUN_TEST(test_authorization_header_matches_aws_vector);
  RUN_TEST(test_canonical_request_sorts_unordered_headers);
  return UNITY_END();
}

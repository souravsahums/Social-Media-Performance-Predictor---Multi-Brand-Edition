"""
API Test Suite — Social Media Performance Predictor

Tests all 20 scenarios from TestCase.md against the live API endpoints.
Also tests health, brands, drift, feedback, and SLM endpoints.

Usage:
    # Start the server first:
    #   uvicorn webapi.main:app --port 8000
    
    # Run all tests:
    python -m pytest test/test_api.py -v
    
    # Run with summary output:
    python -m pytest test/test_api.py -v --tb=short

    # Run standalone (without pytest):
    python test/test_api.py
"""

import sys
import time
import json
import requests
import pytest

BASE_URL = "http://localhost:8000"

# ═══════════════════════════════════════════════════════════════════════
# Test Cases (from TestCase.md)
# ═══════════════════════════════════════════════════════════════════════

TEST_CASES = [
    {
        "id": 1,
        "name": "High-Performance Reel + Celebrity Collab",
        "expected": "high",
        "payload": {
            "metadata_content": {
                "caption": "Who's ready to feel the thunder? ⚡ Drop your city in the comments and win a Thums Up hamper! Tag 3 friends who need this energy 🔥 #ThumsUp #ToofaniEnergy #Giveaway",
                "media_name": "reel",
                "duration": 22,
                "is_collaborated_post": True,
                "collaborators": ["virat.kohli"],
                "created_at": "2026-07-15T19:00:00"
            },
            "profile_stats": {
                "username": "thumsupofficial",
                "followers": 350000,
                "post_author_username": "virat.kohli"
            },
            "media": [{"type": "video", "summary": "Virat Kohli drinking Thums Up, dramatic slow motion, stadium background, brand logo prominent"}]
        }
    },
    {
        "id": 2,
        "name": "Low-Performance Static Post (Minimal Effort)",
        "expected": "low",
        "payload": {
            "metadata_content": {
                "caption": "New.",
                "media_name": "post",
                "duration": 0,
                "is_collaborated_post": False,
                "collaborators": [],
                "created_at": "2026-03-10T03:00:00"
            },
            "profile_stats": {
                "username": "pepsiindia",
                "followers": 1200000,
                "post_author_username": "pepsiindia"
            },
            "media": [{"type": "thumbnail", "summary": "product can on plain white background"}]
        }
    },
    {
        "id": 3,
        "name": "Average Branded Reel — Medium",
        "expected": "medium",
        "payload": {
            "metadata_content": {
                "caption": "The taste that keeps you going through Monday blues. Refresh your mood with every sip 🍋 #Sprite #MondayMotivation #StayCool",
                "media_name": "reel",
                "duration": 45,
                "is_collaborated_post": False,
                "collaborators": [],
                "created_at": "2026-05-12T12:00:00"
            },
            "profile_stats": {
                "username": "sprite_india",
                "followers": 170000,
                "post_author_username": "sprite_india"
            },
            "media": [{"type": "video", "summary": "bottle pouring into glass with ice cubes, product focus, brand colors"}]
        }
    },
    {
        "id": 4,
        "name": "Album with Educational Caption",
        "expected": "medium",
        "payload": {
            "metadata_content": {
                "caption": "5 things you didn't know about Red Bull 🤯\n\n1. Founded in 1987 in Austria\n2. Sold in 175 countries\n3. Sponsors 600+ athletes worldwide\n4. Owns 2 F1 teams\n5. Gives you wings since '97\n\nWhich fact surprised you? Tell us below! 👇\n\n#RedBull #DidYouKnow #EnergyDrink #F1 #Sports #GivesYouWings",
                "media_name": "album",
                "duration": 0,
                "is_collaborated_post": False,
                "collaborators": [],
                "created_at": "2026-08-20T18:30:00"
            },
            "profile_stats": {
                "username": "redbullindia",
                "followers": 1800000,
                "post_author_username": "redbullindia"
            },
            "media": [{"type": "thumbnail", "summary": "infographic with Red Bull facts, bold text on dark background"}]
        }
    },
    {
        "id": 5,
        "name": "Empty Caption Edge Case",
        "expected": "medium",
        "payload": {
            "metadata_content": {
                "caption": "",
                "media_name": "reel",
                "duration": 15,
                "is_collaborated_post": False,
                "collaborators": [],
                "created_at": "2026-06-01T20:00:00"
            },
            "profile_stats": {
                "username": "cocacola_india",
                "followers": 2500000,
                "post_author_username": "cocacola_india"
            },
            "media": [{"type": "video", "summary": "people dancing at a party with Coca-Cola bottles, colorful lights"}]
        }
    },
    {
        "id": 6,
        "name": "Extremely Long Video (180s)",
        "expected": "medium",
        "payload": {
            "metadata_content": {
                "caption": "Full BTS of our latest campaign! From concept to final cut — watch the complete journey 🎬✨ Every frame tells a story. #BTS #CocaCola #MakingOf #ContentCreation #BrandCampaign",
                "media_name": "reel",
                "duration": 180,
                "is_collaborated_post": True,
                "collaborators": ["director_kumar", "dop_sharma"],
                "created_at": "2026-09-10T17:00:00"
            },
            "profile_stats": {
                "username": "cocacola_india",
                "followers": 2500000,
                "post_author_username": "director_kumar"
            },
            "media": [{"type": "video", "summary": "behind the scenes footage, camera equipment, studio setup, team working"}]
        }
    },
    {
        "id": 7,
        "name": "Weekend UGC + Multiple Collabs",
        "expected": "high",
        "payload": {
            "metadata_content": {
                "caption": "When the fam comes together for Thums Up! 🤘⚡ Shot by our amazing community. Keep the Toofani content coming!\n\nTag us and use #ThumsUpFam for a chance to be featured!\n\n📸 @adventure_junkie @mumbai_explorer",
                "media_name": "reel",
                "duration": 28,
                "is_collaborated_post": True,
                "collaborators": ["adventure_junkie", "mumbai_explorer"],
                "created_at": "2026-10-05T19:30:00"
            },
            "profile_stats": {
                "username": "thumsupofficial",
                "followers": 350000,
                "post_author_username": "adventure_junkie"
            },
            "media": [{"type": "video", "summary": "group of young people doing adventure activities, wearing branded gear, mountain backdrop"}]
        }
    },
    {
        "id": 8,
        "name": "Unknown Brand (Not in Training)",
        "expected": "medium",
        "payload": {
            "metadata_content": {
                "caption": "Introducing our brand new flavor — Mango Madness! Available now 🥭🔥 Who's trying it first?",
                "media_name": "reel",
                "duration": 20,
                "is_collaborated_post": False,
                "collaborators": [],
                "created_at": "2026-04-01T18:00:00"
            },
            "profile_stats": {
                "username": "fanta_india",
                "followers": 50000,
                "post_author_username": "fanta_india"
            },
            "media": [{"type": "video", "summary": "mango being sliced, juice splashing, product bottle reveal at end"}]
        }
    },
    {
        "id": 9,
        "name": "Zero Followers (New Account)",
        "expected": "low",
        "payload": {
            "metadata_content": {
                "caption": "Welcome to our official page! 🎉 Follow us for exciting content and giveaways. Hit that follow button! 🚀",
                "media_name": "post",
                "duration": 0,
                "is_collaborated_post": False,
                "collaborators": [],
                "created_at": "2026-01-15T10:00:00"
            },
            "profile_stats": {
                "username": "newbrand_india",
                "followers": 0,
                "post_author_username": "newbrand_india"
            },
            "media": [{"type": "thumbnail", "summary": "welcome message with brand logo, colorful design"}]
        }
    },
    {
        "id": 10,
        "name": "Red Bull Extreme Sports",
        "expected": "high",
        "payload": {
            "metadata_content": {
                "caption": "Gravity is just a suggestion 🪂\n\nWatch @skydiver_raj push beyond limits at 15,000 feet. Would you try this? Comment YES or NO!\n\n#RedBull #GivesYouWings #Skydiving #ExtremeSports #Adventure",
                "media_name": "reel",
                "duration": 18,
                "is_collaborated_post": True,
                "collaborators": ["skydiver_raj"],
                "created_at": "2026-11-20T18:00:00"
            },
            "profile_stats": {
                "username": "redbullindia",
                "followers": 1800000,
                "post_author_username": "skydiver_raj"
            },
            "media": [{"type": "video", "summary": "Person skydiving, GoPro angle, clouds visible, Red Bull logo on helmet"}]
        }
    },
    {
        "id": 11,
        "name": "Pepsi Celebrity Campaign",
        "expected": "high",
        "payload": {
            "metadata_content": {
                "caption": "Jab passion meets swag, magic happens 💙⚡\n\n@ranveersingh knows what it takes. Swag se karenge sabka swagat!\n\nNew TVC dropping tomorrow. Stay tuned! 📺\n\n#Pepsi #SwagStepChallenge #RanveerSingh",
                "media_name": "reel",
                "duration": 32,
                "is_collaborated_post": True,
                "collaborators": ["ranveersingh"],
                "created_at": "2026-03-25T20:00:00"
            },
            "profile_stats": {
                "username": "pepsiindia",
                "followers": 1200000,
                "post_author_username": "ranveersingh"
            },
            "media": [{"type": "video", "summary": "Ranveer Singh dancing with Pepsi can, high production value, multiple camera angles"}]
        }
    },
    {
        "id": 12,
        "name": "Lazy Content (Bare Minimum)",
        "expected": "low",
        "payload": {
            "metadata_content": {
                "caption": "#ad #sponsored",
                "media_name": "post",
                "duration": 0,
                "is_collaborated_post": False,
                "collaborators": [],
                "created_at": "2026-02-14T14:00:00"
            },
            "profile_stats": {
                "username": "cocacola_india",
                "followers": 2500000,
                "post_author_username": "cocacola_india"
            },
            "media": [{"type": "thumbnail", "summary": "plain product packaging image, no people, white background"}]
        }
    },
    {
        "id": 13,
        "name": "Morning Motivational Poll",
        "expected": "medium",
        "payload": {
            "metadata_content": {
                "caption": "Rise and grind! 🌅💪\n\nEvery morning is a fresh start. What's your energy booster?\n\nA) Red Bull 🥫\nB) Coffee ☕\nC) Pure willpower 💪\n\nComment below! #MorningMotivation #RedBull #EnergyBoost #MondayMood",
                "media_name": "reel",
                "duration": 12,
                "is_collaborated_post": False,
                "collaborators": [],
                "created_at": "2026-06-22T07:00:00"
            },
            "profile_stats": {
                "username": "redbullindia",
                "followers": 1800000,
                "post_author_username": "redbullindia"
            },
            "media": [{"type": "video", "summary": "sunrise time-lapse, person stretching, motivational text overlay"}]
        }
    },
    {
        "id": 14,
        "name": "Diwali Festival Campaign",
        "expected": "medium",
        "payload": {
            "metadata_content": {
                "caption": "This Diwali, open happiness with everyone you love 🪔✨🎆\n\nShare this reel with someone who makes your celebrations brighter! Double tap if you're excited 🎉\n\n#CocaCola #OpenHappiness #Diwali2026 #FestivalVibes #DiwaliWithCoke",
                "media_name": "reel",
                "duration": 30,
                "is_collaborated_post": False,
                "collaborators": [],
                "created_at": "2026-10-29T19:00:00"
            },
            "profile_stats": {
                "username": "cocacola_india",
                "followers": 2500000,
                "post_author_username": "cocacola_india"
            },
            "media": [{"type": "video", "summary": "family gathering for Diwali dinner, diyas lit, Coca-Cola bottles on table, warm colors, people smiling"}]
        }
    },
    {
        "id": 15,
        "name": "Spam Hashtags (Over-optimization)",
        "expected": "medium",
        "payload": {
            "metadata_content": {
                "caption": "Summer vibes 🌊 #pepsi #cola #summer #drink #refreshing #cold #ice #cool #party #fun #friends #beach #sun #hot #trending #viral #explore #fyp #foryou #instagram #reels #india #mumbai #delhi #bangalore",
                "media_name": "reel",
                "duration": 25,
                "is_collaborated_post": False,
                "collaborators": [],
                "created_at": "2026-05-20T18:00:00"
            },
            "profile_stats": {
                "username": "pepsiindia",
                "followers": 1200000,
                "post_author_username": "pepsiindia"
            },
            "media": [{"type": "video", "summary": "person at beach holding Pepsi can, sunset background"}]
        }
    },
    {
        "id": 16,
        "name": "Gen Z Meme / Humor Content",
        "expected": "high",
        "payload": {
            "metadata_content": {
                "caption": "POV: Your friend says 'garmi nahi lag rahi' in 45°C 😂💀\n\nTag that friend who's always in denial! We'll send them a Sprite 🧊\n\n#Sprite #GarmiKaAntidote #SummerMemes #GenZ #RelatableContent",
                "media_name": "reel",
                "duration": 15,
                "is_collaborated_post": False,
                "collaborators": [],
                "created_at": "2026-06-05T20:30:00"
            },
            "profile_stats": {
                "username": "sprite_india",
                "followers": 170000,
                "post_author_username": "sprite_india"
            },
            "media": [{"type": "video", "summary": "split screen meme format, person sweating then drinking Sprite, comedic expressions"}]
        }
    },
    {
        "id": 17,
        "name": "Very Short Duration (5s)",
        "expected": "low",
        "payload": {
            "metadata_content": {
                "caption": "Quick sip! ⚡",
                "media_name": "reel",
                "duration": 5,
                "is_collaborated_post": False,
                "collaborators": [],
                "created_at": "2026-07-10T15:00:00"
            },
            "profile_stats": {
                "username": "cocacola_india",
                "followers": 2500000,
                "post_author_username": "cocacola_india"
            },
            "media": [{"type": "video", "summary": "quick product shot, Coke being poured"}]
        }
    },
    {
        "id": 18,
        "name": "4-Creator Mega Collaboration",
        "expected": "high",
        "payload": {
            "metadata_content": {
                "caption": "When India's biggest creators come together for one mission — SPREAD THE ENERGY! ⚡🇮🇳\n\nWatch the full episode on YouTube (link in bio) 🎬\n\nWho's your favorite creator? Comment below! 👇\n\n@tech_burner @ashish.chanchlani @bhuvan.bam @kusha.kapila",
                "media_name": "reel",
                "duration": 28,
                "is_collaborated_post": True,
                "collaborators": ["tech_burner", "ashish.chanchlani", "bhuvan.bam", "kusha.kapila"],
                "created_at": "2026-12-01T19:00:00"
            },
            "profile_stats": {
                "username": "redbullindia",
                "followers": 1800000,
                "post_author_username": "tech_burner"
            },
            "media": [{"type": "video", "summary": "group of creators doing extreme challenges, Red Bull branding, high energy atmosphere"}]
        }
    },
    {
        "id": 19,
        "name": "Nostalgic / Emotional Content",
        "expected": "medium",
        "payload": {
            "metadata_content": {
                "caption": "Remember when Sundays meant this? 🏏☀️🍕\n\nGully cricket, friends, and chilled Coke. Some things never change.\n\nTag your gully cricket squad! 👇 #CocaCola #GullyCricket #Nostalgia #90sKids #SundayVibes",
                "media_name": "reel",
                "duration": 25,
                "is_collaborated_post": False,
                "collaborators": [],
                "created_at": "2026-08-18T11:00:00"
            },
            "profile_stats": {
                "username": "cocacola_india",
                "followers": 2500000,
                "post_author_username": "cocacola_india"
            },
            "media": [{"type": "video", "summary": "kids playing cricket in gully, vintage filter, Coca-Cola glass bottle, family watching from balcony"}]
        }
    },
    {
        "id": 20,
        "name": "All Zeros / Robustness Test",
        "expected": "low",
        "payload": {
            "metadata_content": {
                "caption": "",
                "media_name": "post",
                "duration": 0,
                "is_collaborated_post": False,
                "collaborators": [],
                "created_at": ""
            },
            "profile_stats": {
                "username": "unknown_brand",
                "followers": 0,
                "post_author_username": "unknown_brand"
            },
            "media": []
        }
    },
]


# ═══════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════

def _check_server():
    """Check if the API server is running."""
    try:
        r = requests.get(f"{BASE_URL}/api/health", timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


# ═══════════════════════════════════════════════════════════════════════
# Pytest Fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="session", autouse=True)
def check_server():
    """Ensure the API server is running before tests."""
    if not _check_server():
        pytest.skip("API server not running at localhost:8000. Start with: uvicorn webapi.main:app --port 8000")


# ═══════════════════════════════════════════════════════════════════════
# Test: Health Endpoint
# ═══════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    def test_health_returns_200(self):
        r = requests.get(f"{BASE_URL}/api/health")
        assert r.status_code == 200

    def test_health_model_loaded(self):
        r = requests.get(f"{BASE_URL}/api/health")
        data = r.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_health_brands_available(self):
        r = requests.get(f"{BASE_URL}/api/health")
        data = r.json()
        assert len(data["brands_available"]) == 5
        expected_brands = {"cocacola_india", "redbullindia", "pepsiindia",
                           "sprite_india", "thumsupofficial"}
        assert set(data["brands_available"]) == expected_brands


# ═══════════════════════════════════════════════════════════════════════
# Test: Brands Endpoint
# ═══════════════════════════════════════════════════════════════════════

class TestBrandsEndpoint:
    def test_list_brands(self):
        r = requests.get(f"{BASE_URL}/api/brands")
        assert r.status_code == 200
        data = r.json()
        assert "brands" in data
        assert len(data["brands"]) == 5

    def test_brand_stats_valid(self):
        r = requests.get(f"{BASE_URL}/api/brand/sprite_india")
        assert r.status_code == 200
        data = r.json()
        assert data["brand"] == "sprite_india"
        assert "stats" in data
        assert "mean" in data["stats"]
        assert "median" in data["stats"]

    def test_brand_stats_not_found(self):
        r = requests.get(f"{BASE_URL}/api/brand/nonexistent_brand")
        assert r.status_code == 404


# ═══════════════════════════════════════════════════════════════════════
# Test: Predict Endpoint (JSON — /api/predict)
# ═══════════════════════════════════════════════════════════════════════

class TestPredictEndpoint:
    """Tests the /api/predict endpoint with all 20 test cases."""

    @pytest.mark.parametrize("tc", TEST_CASES, ids=[f"TC{t['id']:02d}_{t['name'][:30]}" for t in TEST_CASES])
    def test_predict_returns_valid_response(self, tc):
        """Each test case should return HTTP 200 with valid prediction structure."""
        r = requests.post(f"{BASE_URL}/api/predict", json=tc["payload"])
        assert r.status_code == 200, f"TC{tc['id']}: HTTP {r.status_code} — {r.text}"

        data = r.json()
        # Validate response structure
        assert "prediction" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert data["prediction"] in ("low", "medium", "high")
        assert 0.0 <= data["confidence"] <= 1.0
        assert set(data["probabilities"].keys()) == {"low", "medium", "high"}

    @pytest.mark.parametrize("tc", TEST_CASES, ids=[f"TC{t['id']:02d}_{t['name'][:30]}" for t in TEST_CASES])
    def test_predict_has_explanation(self, tc):
        """Each prediction should include explanation factors."""
        r = requests.post(f"{BASE_URL}/api/predict", json=tc["payload"])
        data = r.json()
        assert "explanation" in data
        assert isinstance(data["explanation"], list)
        assert len(data["explanation"]) > 0

    @pytest.mark.parametrize("tc", TEST_CASES, ids=[f"TC{t['id']:02d}_{t['name'][:30]}" for t in TEST_CASES])
    def test_predict_has_model_predictions(self, tc):
        """Each prediction should include individual model predictions (ensemble)."""
        r = requests.post(f"{BASE_URL}/api/predict", json=tc["payload"])
        data = r.json()
        assert "model_predictions" in data
        mp = data["model_predictions"]
        assert "neural_network" in mp
        assert "random_forest" in mp
        assert mp["neural_network"]["prediction"] in ("low", "medium", "high")
        assert mp["random_forest"]["prediction"] in ("low", "medium", "high")


# ═══════════════════════════════════════════════════════════════════════
# Test: Predict Simple Endpoint (Form — /api/predict-simple)
# ═══════════════════════════════════════════════════════════════════════

class TestPredictSimpleEndpoint:
    def test_basic_prediction(self):
        r = requests.post(f"{BASE_URL}/api/predict-simple", data={
            "brand": "pepsiindia",
            "content_type": "reel",
            "caption": "Summer vibes! #pepsi",
            "duration": "25",
            "is_collab": "true",
            "followers": "500000",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["prediction"] in ("low", "medium", "high")
        assert "confidence" in data
        assert "slm_prediction" in data

    def test_with_thumbnail_url(self):
        r = requests.post(f"{BASE_URL}/api/predict-simple", data={
            "brand": "sprite_india",
            "caption": "Fresh look! 🧊",
            "media_type": "reel",
            "duration": "15",
            "is_collab": "false",
            "followers": "170000",
            "thumbnail_url": "https://consuma-media-public.s3.us-east-1.amazonaws.com/media/thumbnail/2024-02-02_09-02-15_UTC.jpg",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["prediction"] in ("low", "medium", "high")

    def test_drift_warning_present(self):
        """Predictions on new data should include drift warning."""
        r = requests.post(f"{BASE_URL}/api/predict-simple", data={
            "brand": "sprite_india",
            "caption": "Test",
            "media_type": "reel",
            "duration": "30",
            "is_collab": "false",
            "followers": "170000",
        })
        data = r.json()
        assert "drift_warning" in data
        assert data["drift_warning"] in ("low", "medium", "high")

    def test_slm_reasoning_present(self):
        """SLM reasoning should be included in simple predictions."""
        r = requests.post(f"{BASE_URL}/api/predict-simple", data={
            "brand": "redbullindia",
            "caption": "Extreme! 🪂 #redbull",
            "media_type": "reel",
            "duration": "20",
            "is_collab": "true",
            "collaborators": "athlete_1",
            "followers": "1800000",
        })
        data = r.json()
        assert "slm_prediction" in data
        assert "slm_reasoning" in data
        assert data["slm_prediction"] in ("low", "medium", "high")
        assert isinstance(data["slm_reasoning"], list)


# ═══════════════════════════════════════════════════════════════════════
# Test: SLM Endpoint (/api/predict-slm)
# ═══════════════════════════════════════════════════════════════════════

class TestSLMEndpoint:
    def test_slm_basic(self):
        r = requests.post(f"{BASE_URL}/api/predict-slm", data={
            "brand": "sprite_india",
            "caption": "Garmi ka antidote! 🧊 #sprite",
            "media_type": "reel",
            "duration": "25",
            "is_collab": "false",
            "followers": "170000",
        })
        assert r.status_code == 200
        data = r.json()
        assert "prediction" in data
        assert "score" in data
        assert "reasoning" in data
        assert data["prediction"] in ("low", "medium", "high")

    def test_slm_reasoning_has_factors(self):
        r = requests.post(f"{BASE_URL}/api/predict-slm", data={
            "brand": "pepsiindia",
            "caption": "Tag your squad! 🎉 #pepsi #summer",
            "media_type": "reel",
            "duration": "30",
            "is_collab": "true",
            "collaborators": "creator1",
            "followers": "1200000",
        })
        data = r.json()
        assert len(data["reasoning"]) > 0
        for factor in data["reasoning"]:
            assert "factor" in factor
            assert "points" in factor
            assert "explanation" in factor

    def test_slm_collab_scores_high(self):
        """Reel + collab + optimal duration should score high in SLM."""
        r = requests.post(f"{BASE_URL}/api/predict-slm", data={
            "brand": "thumsupofficial",
            "caption": "Thunder! ⚡ Tag 3 friends! #ThumsUp",
            "media_type": "reel",
            "duration": "22",
            "is_collab": "true",
            "collaborators": "celebrity",
            "followers": "350000",
            "visual_summary": "person drinking, brand visible",
        })
        data = r.json()
        assert data["prediction"] == "high"
        assert data["score"] >= 65


# ═══════════════════════════════════════════════════════════════════════
# Test: Drift Monitoring Endpoints
# ═══════════════════════════════════════════════════════════════════════

class TestDriftEndpoints:
    def test_drift_status(self):
        r = requests.get(f"{BASE_URL}/api/drift/status")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data or "message" in data

    def test_drift_check(self):
        r = requests.get(f"{BASE_URL}/api/drift/check")
        assert r.status_code == 200
        data = r.json()
        assert "message" in data or "risk_level" in data


# ═══════════════════════════════════════════════════════════════════════
# Test: Feedback Endpoints
# ═══════════════════════════════════════════════════════════════════════

class TestFeedbackEndpoints:
    def test_feedback_summary(self):
        r = requests.get(f"{BASE_URL}/api/feedback/summary")
        assert r.status_code == 200
        data = r.json()
        assert "total_entries" in data or "total" in data or isinstance(data, dict)

    def test_retraining_pipeline(self):
        r = requests.get(f"{BASE_URL}/api/feedback/retraining-pipeline")
        assert r.status_code == 200

    def test_submit_feedback(self):
        """Submit a feedback correction."""
        r = requests.post(f"{BASE_URL}/api/feedback", json={
            "prediction_id": "test_001",
            "predicted_label": "low",
            "correct_label": "high",
            "features": {"duration": 25, "is_reel": 1},
            "brand": "sprite_india",
            "caption": "Test caption",
            "media_type": "reel",
        })
        assert r.status_code == 200
        data = r.json()
        assert "status" in data


# ═══════════════════════════════════════════════════════════════════════
# Test: Evaluation Endpoint
# ═══════════════════════════════════════════════════════════════════════

class TestEvaluationEndpoint:
    def test_evaluation_results(self):
        r = requests.get(f"{BASE_URL}/api/evaluation")
        # May return 404 if evaluation hasn't been run
        assert r.status_code in (200, 404)


# ═══════════════════════════════════════════════════════════════════════
# Test: Response Latency
# ═══════════════════════════════════════════════════════════════════════

class TestPerformance:
    def test_prediction_latency(self):
        """Prediction should complete within 500ms (without thumbnail download)."""
        payload = TEST_CASES[0]["payload"]
        start = time.time()
        r = requests.post(f"{BASE_URL}/api/predict", json=payload)
        elapsed = time.time() - start
        assert r.status_code == 200
        assert elapsed < 0.5, f"Prediction took {elapsed:.2f}s (>500ms threshold)"

    def test_health_latency(self):
        """Health endpoint should respond within 100ms."""
        start = time.time()
        r = requests.get(f"{BASE_URL}/api/health")
        elapsed = time.time() - start
        assert r.status_code == 200
        assert elapsed < 0.1, f"Health check took {elapsed:.2f}s (>100ms threshold)"


# ═══════════════════════════════════════════════════════════════════════
# Test: Edge Cases & Error Handling
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_payload(self):
        """Empty JSON body should return 422 (validation error)."""
        r = requests.post(f"{BASE_URL}/api/predict", json={})
        assert r.status_code == 422

    def test_invalid_brand_in_predict(self):
        """Unknown brand should still return a prediction (graceful fallback)."""
        r = requests.post(f"{BASE_URL}/api/predict", json={
            "metadata_content": {
                "caption": "Test",
                "media_name": "reel",
                "duration": 15,
            },
            "profile_stats": {
                "username": "completely_unknown_brand",
                "followers": 1000,
            },
            "media": [],
        })
        assert r.status_code == 200
        data = r.json()
        assert data["prediction"] in ("low", "medium", "high")

    def test_negative_duration_rejected(self):
        """Negative duration should be rejected by validation."""
        r = requests.post(f"{BASE_URL}/api/predict", json={
            "metadata_content": {
                "caption": "Test",
                "media_name": "reel",
                "duration": -10,
            },
            "profile_stats": {
                "username": "sprite_india",
                "followers": 170000,
            },
            "media": [],
        })
        assert r.status_code == 422

    def test_very_large_followers(self):
        """Extremely large follower count should not crash."""
        r = requests.post(f"{BASE_URL}/api/predict", json={
            "metadata_content": {
                "caption": "Global brand post",
                "media_name": "reel",
                "duration": 30,
            },
            "profile_stats": {
                "username": "cocacola_india",
                "followers": 100000000,
            },
            "media": [],
        })
        assert r.status_code == 200

    def test_unicode_caption(self):
        """Unicode/emoji-heavy caption should work fine."""
        r = requests.post(f"{BASE_URL}/api/predict", json={
            "metadata_content": {
                "caption": "🔥🔥🔥 गर्मी का इलाज! 🧊❄️ \n\n#हिंदी #ट्रेंडिंग 🇮🇳",
                "media_name": "reel",
                "duration": 20,
            },
            "profile_stats": {
                "username": "sprite_india",
                "followers": 170000,
            },
            "media": [],
        })
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# Standalone Runner (without pytest)
# ═══════════════════════════════════════════════════════════════════════

def run_standalone():
    """Run tests without pytest and generate a results summary."""
    if not _check_server():
        print("ERROR: API server not running at localhost:8000")
        print("Start with: uvicorn webapi.main:app --port 8000")
        sys.exit(1)

    print("=" * 70)
    print("SOCIAL MEDIA PERFORMANCE PREDICTOR — API TEST SUITE")
    print("=" * 70)
    print(f"\nServer: {BASE_URL}")
    print(f"Test cases: {len(TEST_CASES)}\n")

    # Health check
    print("─" * 70)
    print("1. HEALTH & SYSTEM ENDPOINTS")
    print("─" * 70)
    r = requests.get(f"{BASE_URL}/api/health")
    health = r.json()
    print(f"  ✓ Health: {health['status']} | Model: {'loaded' if health['model_loaded'] else 'NOT LOADED'}")
    print(f"  ✓ Brands: {', '.join(health['brands_available'])}")

    r = requests.get(f"{BASE_URL}/api/brands")
    print(f"  ✓ Brands endpoint: {len(r.json()['brands'])} brands returned")

    r = requests.get(f"{BASE_URL}/api/drift/status")
    print(f"  ✓ Drift status: {r.json().get('status', r.json().get('message', 'ok'))}")

    # Prediction tests
    print(f"\n{'─' * 70}")
    print("2. PREDICTION TESTS (20 Test Cases)")
    print("─" * 70)
    print(f"\n  {'#':<4} {'Test Case':<40} {'Expected':<10} {'NN+RF':<10} {'SLM':<10} {'Match'}")
    print(f"  {'─'*4} {'─'*40} {'─'*10} {'─'*10} {'─'*10} {'─'*5}")

    nn_matches = 0
    slm_matches = 0
    total_latency = 0
    results = []

    for tc in TEST_CASES:
        start = time.time()
        r = requests.post(f"{BASE_URL}/api/predict", json=tc["payload"])
        elapsed = time.time() - start
        total_latency += elapsed

        if r.status_code != 200:
            print(f"  {tc['id']:<4} {tc['name']:<40} {'ERROR':>10}")
            results.append({"id": tc["id"], "status": "error"})
            continue

        data = r.json()
        ensemble_pred = data["prediction"]
        nn_pred = data.get("model_predictions", {}).get("neural_network", {}).get("prediction", "?")

        # Get SLM prediction
        slm_r = requests.post(f"{BASE_URL}/api/predict-slm", data={
            "brand": tc["payload"]["profile_stats"]["username"],
            "caption": tc["payload"]["metadata_content"]["caption"],
            "media_type": tc["payload"]["metadata_content"]["media_name"],
            "duration": str(tc["payload"]["metadata_content"]["duration"]),
            "is_collab": str(tc["payload"]["metadata_content"]["is_collaborated_post"]).lower(),
            "collaborators": ",".join(tc["payload"]["metadata_content"].get("collaborators", [])),
            "followers": str(tc["payload"]["profile_stats"]["followers"]),
            "visual_summary": tc["payload"].get("media", [{}])[0].get("summary", "") if tc["payload"].get("media") else "",
        })
        slm_pred = slm_r.json().get("prediction", "?") if slm_r.status_code == 200 else "err"

        expected = tc["expected"]
        ens_match = "✓" if ensemble_pred == expected else "✗"
        slm_match_flag = "✓" if slm_pred == expected else "✗"

        if ensemble_pred == expected:
            nn_matches += 1
        if slm_pred == expected:
            slm_matches += 1

        print(f"  {tc['id']:<4} {tc['name']:<40} {expected:<10} {ensemble_pred:<10} {slm_pred:<10} {ens_match}")

        results.append({
            "id": tc["id"],
            "name": tc["name"],
            "expected": expected,
            "ensemble": ensemble_pred,
            "nn": nn_pred,
            "slm": slm_pred,
            "confidence": data.get("confidence", 0),
            "latency_ms": round(elapsed * 1000, 1),
        })

    # Summary
    print(f"\n{'─' * 70}")
    print("3. SUMMARY")
    print("─" * 70)
    print(f"\n  Ensemble (NN+RF) match with expected: {nn_matches}/{len(TEST_CASES)} ({nn_matches/len(TEST_CASES)*100:.0f}%)")
    print(f"  SLM match with expected:              {slm_matches}/{len(TEST_CASES)} ({slm_matches/len(TEST_CASES)*100:.0f}%)")
    print(f"  Average latency:                      {total_latency/len(TEST_CASES)*1000:.0f}ms per prediction")
    print(f"  All responses HTTP 200:               {'Yes' if all(r.get('status') != 'error' for r in results) else 'No'}")

    # Edge case tests
    print(f"\n{'─' * 70}")
    print("4. EDGE CASE & ERROR HANDLING TESTS")
    print("─" * 70)

    # Empty payload
    r = requests.post(f"{BASE_URL}/api/predict", json={})
    print(f"  {'✓' if r.status_code == 422 else '✗'} Empty payload returns 422: {r.status_code}")

    # Negative duration
    r = requests.post(f"{BASE_URL}/api/predict", json={
        "metadata_content": {"caption": "x", "media_name": "reel", "duration": -1},
        "profile_stats": {"username": "sprite_india", "followers": 100},
        "media": [],
    })
    print(f"  {'✓' if r.status_code == 422 else '✗'} Negative duration returns 422: {r.status_code}")

    # Unknown brand graceful
    r = requests.post(f"{BASE_URL}/api/predict", json={
        "metadata_content": {"caption": "test", "media_name": "reel", "duration": 15},
        "profile_stats": {"username": "totally_unknown", "followers": 1000},
        "media": [],
    })
    print(f"  {'✓' if r.status_code == 200 else '✗'} Unknown brand returns 200: {r.status_code}")

    # Unicode
    r = requests.post(f"{BASE_URL}/api/predict", json={
        "metadata_content": {"caption": "🔥गर्मी🧊 #हिंदी", "media_name": "reel", "duration": 20},
        "profile_stats": {"username": "sprite_india", "followers": 170000},
        "media": [],
    })
    print(f"  {'✓' if r.status_code == 200 else '✗'} Unicode caption returns 200: {r.status_code}")

    # Large followers
    r = requests.post(f"{BASE_URL}/api/predict", json={
        "metadata_content": {"caption": "big", "media_name": "reel", "duration": 30},
        "profile_stats": {"username": "cocacola_india", "followers": 100000000},
        "media": [],
    })
    print(f"  {'✓' if r.status_code == 200 else '✗'} Large followers returns 200: {r.status_code}")

    print(f"\n{'═' * 70}")
    print("TEST SUITE COMPLETE")
    print(f"{'═' * 70}\n")

    # Save results to JSON
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "server": BASE_URL,
        "total_test_cases": len(TEST_CASES),
        "ensemble_matches": nn_matches,
        "slm_matches": slm_matches,
        "avg_latency_ms": round(total_latency / len(TEST_CASES) * 1000, 1),
        "results": results,
    }
    with open("artifacts/test_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to artifacts/test_results.json")


if __name__ == "__main__":
    run_standalone()

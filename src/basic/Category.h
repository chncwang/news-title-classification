#ifndef NEWS_SRC_BASIC_CATEGORY_H
#define NEWS_SRC_BASIC_CATEGORY_H

#include <string>

enum Category {
    BABY,
    BEAUTY,
    CAR,
    COMIC,
    CONSTELLATION,
    CULTURAL,
    DESIGN,
    DIGI,
    DRAMA,
    DRESS,
    EDUCATION,
    FOOD,
    GAME,
    HEALTH,
    HOUSE,
    IT,
    JOKE,
    LOTTERY,
    MANAGE,
    MASS_COMMUNICATION,
    MONEY,
    MOVIE,
    MUSIC,
    NEWS,
    PET,
    PHOTO,
    SCIENCE,
    SEX,
    SPORTS,
    STAR,
    TRAVEL,
    TV
};

Category ToCategory(const std::string &str) {
    if (str == "baby") {
        return Category::BABY;
    } else if (str == "beauty") {
        return Category::BEAUTY;
    } else if (str == "car") {
        return Category::CAR;
    } else if (str == "comic") {
        return Category::COMIC;
    } else if (str == "constellation") {
        return Category::CONSTELLATION;
    } else if (str == "cultural") {
        return Category::CULTURAL;
    } else if (str == "design") {
        return Category::DESIGN;
    } else if (str == "digi") {
        return Category::DIGI;
    } else if (str == "drama") {
        return Category::DRAMA;
    } else if (str == "dress") {
        return Category::DRESS;
    } else if (str == "education") {
        return Category::EDUCATION;
    } else if (str == "food") {
        return Category::FOOD;
    } else if (str == "game") {
        return Category::GAME;
    } else if (str == "health") {
        return Category::HEALTH;
    } else if (str == "house") {
        return Category::HOUSE;
    } else if (str == "it") {
        return Category::IT;
    } else if (str == "joke") {
        return Category::JOKE;
    } else if (str == "lottery") {
        return Category::LOTTERY;
    } else if (str == "manage") {
        return Category::MANAGE;
    } else if (str == "mass_communication") {
        return Category::MASS_COMMUNICATION;
    } else if (str == "money") {
        return Category::MONEY;
    } else if (str == "movie") {
        return Category::MOVIE;
    } else if (str == "music") {
        return Category::MUSIC;
    } else if (str == "news") {
        return Category::NEWS;
    } else if (str == "pet") {
        return Category::PET;
    } else if (str == "photo") {
        return Category::PHOTO;
    } else if (str == "science") {
        return Category::SCIENCE;
    } else if (str == "sex") {
        return Category::SEX;
    } else if (str == "sports") {
        return Category::SPORTS;
    } else if (str == "star") {
        return Category::STAR;
    } else if (str == "travel") {
        return Category::TRAVEL;
    } else if (str == "tv") {
        return Category::TV;
    } else {
        abort();
    }
}

#endif

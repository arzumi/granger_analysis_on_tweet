from User.UserManager import UserManager
from User.UserType import UserType
from Tweet.TweetType import TweetType
from Aggregation.AggregationBase import AggregationBase
from Mapping.ContentType import ContentType
from Tweet.ContentSpaceTweet import ContentSpaceTweet
from Tweet.MinimalTweet import MinimalTweet
from User.ContentSpaceUser import ContentSpaceUser
from Aggregation.ContentSpace import ContentSpace

from typing import Dict, List, Any, Set, DefaultDict, Union
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

def _merge_dicts(dict1: Dict[Any, Set[MinimalTweet]], dict2: Dict[Any,
                 Set[MinimalTweet]]) -> None:
    """Update the value (set) in dict1 by the value (set) in dict2.
    """
    for key, value in dict2.items():
        dict1[key].update(value)


def _clear_by_time_helper(start: datetime, end: datetime, storage:
                          Dict[Union[UserType, int],
                          DefaultDict[Any, Set[MinimalTweet]]]) -> None:
    """A helper function that for the leaf value (set of tweets), only
    retain the tweets whose created time is between <start> and <end>.
    a"""
    for user, map_dict in storage.items():
        for content_type, tweet_set in map_dict.items():
            new_set = {tweet for tweet in tweet_set if start <=
                       tweet.created_at < end}
            storage[user][content_type] = new_set


class ContentDemandSupply(AggregationBase):
    """Aggregate Supply and Demand Information for time series processing.
    """
    # Attributes
    name: str
    content_space: Set[ContentType]
    user_manager: UserManager

    demand_in_community: Dict[Union[UserType, int], DefaultDict[Any, Set[MinimalTweet]]]
    demand_out_community: Dict[Union[UserType, int], DefaultDict[Any, Set[MinimalTweet]]]
    # add retweets of out community by in community
    demand_out_community_by_in_community: Dict[Union[UserType, int], DefaultDict[Any, Set[MinimalTweet]]]
    supply: Dict[Union[UserType, int], DefaultDict[Any, Set[MinimalTweet]]]

    def __init__(self, *args):
        # create()
        # param: str, Set[ContentType], UserManager, TweetManager
        if len(args) == 4:
            super().__init__(args[0], args[2], args[3])
            # load from arguments
            self.content_space = args[1]
            self.user_manager = args[2]

            # initialize demand and supply
            self.demand_in_community = {UserType.CONSUMER: defaultdict(set),
                                        UserType.CORE_NODE: defaultdict(set)}
            for user in self.user_manager.users:
                self.demand_in_community[user.user_id] = defaultdict(set)
            self.demand_out_community = {UserType.CONSUMER: defaultdict(set),
                                         UserType.CORE_NODE: defaultdict(set)}
            for user in self.user_manager.users:
                self.demand_out_community[user.user_id] = defaultdict(set)
            # add retweets of out community by in community
            self.demand_out_community_by_in_community = {UserType.CONSUMER: defaultdict(set),
                                                         UserType.CORE_NODE: defaultdict(set)}
            for user in self.user_manager.users:
                self.demand_out_community_by_in_community[user.user_id] = defaultdict(set)
            self.supply = {UserType.CORE_NODE: defaultdict(set),
                           UserType.PRODUCER: defaultdict(set)}
            for user in self.user_manager.users:
                self.supply[user.user_id] = defaultdict(set)
        # load()
        # param: str, Set[ContentType],
        #        Dict[UserType, Dict[Any, Set[MinimalTweet]]],
        #        Dict[UserType, Dict[Any, Set[MinimalTweet]]]
        elif len(args) == 6:
            self.name = args[0]
            self.content_space = args[1]
            self.demand_in_community = args[2]
            self.demand_out_community = args[3]
            # add retweets of out community by in community
            self.demand_out_community_by_in_community = args[4]
            self.supply = args[5]

    def get_all_content_type_repr(self) -> List[Any]:
        """Return a list of all representation of ContentType.
        """
        return [content_type.get_representation() for content_type
                in self.content_space]

    def _calculate_user_type_mapping(self, user_type: UserType,
                                     storage: Dict[Any,
                                     Dict[Any, Set[MinimalTweet]]],
                                     tweet_types: List[TweetType]) -> None:
        """A helper function that calculate the mapping of tweets with
        <tweet_types> of <user_type> and store in <storage>.
        """
        for user in tqdm(self.user_manager.get_type_users(user_type)):
            # ignore this type warning
            freq_dict = self.user_manager.calculate_user_time_mapping(
                user, tweet_types)
            _merge_dicts(storage[user_type], freq_dict)

    def _calculate_user_mapping(self, user: ContentSpaceUser,
                                storage: Dict[Any,
                                Dict[Any, Set[MinimalTweet]]],
                                tweet_types: List[TweetType]) -> None:
        """A helper function that calculate the mapping of tweets with
        <tweet_types> of <user> and store in <storage>.
        """
        freq_dict = self.user_manager.calculate_user_time_mapping(
                user, tweet_types)
        storage[user.user_id] = freq_dict

    def calculate_demand_in_community(self):
        """Calculate demand in community for UserType and individual user.
        """
        print("==============Start User Demand In Community==============")
        demand_spec = [TweetType.RETWEET_OF_IN_COMM]
        self._calculate_user_type_mapping(UserType.CONSUMER, self.demand_in_community,
                                          demand_spec)
        self._calculate_user_type_mapping(UserType.CORE_NODE, self.demand_in_community,
                                          demand_spec)
        for user in self.user_manager.users:
            self._calculate_user_mapping(user, self.demand_in_community, demand_spec)
        print("=========Successfully Create User Demand In Community=========")

    def calculate_demand_out_community(self):
        """Calculate demand out community for UserType and individual user.
        """
        print("==============Start User Demand Out Community==============")
        demand_spec = [TweetType.RETWEET_OF_OUT_COMM]
        self._calculate_user_type_mapping(UserType.CONSUMER, self.demand_out_community,
                                          demand_spec)
        self._calculate_user_type_mapping(UserType.CORE_NODE, self.demand_out_community,
                                          demand_spec)
        for user in self.user_manager.users:
            self._calculate_user_mapping(user, self.demand_out_community, demand_spec)
        print("=========Successfully Create User Demand Out Community=========")

    # add retweets of out community by in community
    def calculate_demand_out_community_by_in_community(self):
        print("Start User Demand Out Community by In Community")
        demand_spec = [TweetType.RETWEET_OF_OUT_COMM_BY_IN_COMM]
        self._calculate_user_type_mapping(UserType.CONSUMER, self.demand_out_community_by_in_community,
                                          demand_spec)
        self._calculate_user_type_mapping(UserType.CORE_NODE, self.demand_out_community_by_in_community,
                                          demand_spec)
        for user in self.user_manager.users:
            self._calculate_user_mapping(user, self.demand_out_community_by_in_community, demand_spec)

    def calculate_supply(self):
        """Calculate supply for UserType and individual user.
        """
        print("==============Start User Supply==============")
        supply_spec = [TweetType.ORIGINAL_TWEET]
        self._calculate_user_type_mapping(UserType.PRODUCER, self.supply,
                                          supply_spec)
        self._calculate_user_type_mapping(UserType.CORE_NODE, self.supply,
                                          supply_spec)
        for user in self.user_manager.users:
            self._calculate_user_mapping(user, self.supply, supply_spec)
        print("=========Successfully Create User Supply=========")

    def clear_tweets_by_time(self, start: datetime, end: datetime) -> None:
        """Remove the Tweets in ds which create time is not between
        start and end.
        """
        # Note: not used now
        _clear_by_time_helper(start, end, self.demand_in_community)
        _clear_by_time_helper(start, end, self.demand_out_community)
        # add retweets of out community by in community
        _clear_by_time_helper(start, end, self.demand_out_community_by_in_community)
        _clear_by_time_helper(start, end, self.supply)

    def get_tweets_by_type(self, content_type: Any, tweet_type: TweetType,
                           space: ContentSpace) -> Set[ContentSpaceTweet]:
        """Get all the tweets of <tweet_type> with <content_type>.
        """
        # 1. Get all targeted tweets
        if tweet_type == TweetType.ORIGINAL_TWEET:
            tweet_set = (self.supply[UserType.CORE_NODE][content_type] |
                         self.supply[UserType.PRODUCER][content_type])
        elif tweet_type == TweetType.RETWEET_OF_IN_COMM:
            tweet_set = (self.demand_in_community[UserType.CONSUMER][content_type] |
                         self.demand_in_community[UserType.CORE_NODE][content_type])
        elif tweet_type == TweetType.RETWEET_OF_OUT_COMM:
            tweet_set = (self.demand_out_community[UserType.CONSUMER][content_type] |
                         self.demand_out_community[UserType.CORE_NODE][content_type])
        else:
            tweet_set = set()

        # 2. Retreive ContentSpaceTweet instead
        return {space.get_tweet(tweet.id) for tweet in tweet_set}

from UserPartitioning.UserPartitioningStrategy import UserPartitioningStrategy
from User.UserBase import UserBase

# core_node_list = [1330571318971027462, 917892794953404417, 23612012, 3161912605,
#                   227629567, 919900711, 301042394, 228660231, 2233129128,
#                   4369711156, 1884178352, 1651411087, 126345156, 232951413,
#                   277594186, 313299656, 186797066, 92284830, 1729528081, 13247182]

# Choose the top 10
# core_node_list = [23612012, 3161912605,
#                   227629567, 919900711, 301042394, 228660231,
#                   4369711156, 1651411087, 232951413,
#                   277594186]
core_node_list = [228660231]

class SocialSupportStrategy(UserPartitioningStrategy):
    """Classifies user as a producer, consumer, or core node
    """
    def is_consumer(self, user: UserBase) -> bool:
        """If a user is not a core node, then is a consumer.
        """
        return not self.is_core_node(user)

    def is_producer(self, user: UserBase) -> bool:
        """If a user is not a core node, then is a producer.
        """
        return not self.is_core_node(user)

    def is_core_node(self, user: UserBase) -> bool:
        """Return True if a user is ranked top in social support so in
        <core_node_list>.
        """
        return user.user_id in core_node_list
